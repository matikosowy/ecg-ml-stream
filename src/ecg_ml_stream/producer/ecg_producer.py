"""Multi-thread Kafka producer module simulating real-time ECG data.

Copyright 2026 Mateusz Golebiewski
"""

import argparse
import json
import logging
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from kafka import KafkaProducer
from kafka.errors import KafkaError

from ecg_ml_stream.dataset.ecg_dataset import ECGDataset
from ecg_ml_stream.utils.constants import ECG_LEAD_NAMES
from ecg_ml_stream.utils.helpers import setup_logging

logger = logging.getLogger("producer")


class ECGProducer:
    """Multi-thread Kafka producer that simulates hospital ECG streams.

    Each thread represents one hospital and sends one PTB-XL record every `interval_sec`
    seconds to the specified Kafka topic.

    Attributes:
        HOSPITALS (ClassVar[list[dict]]): List of simulated hospitals with IDs

    """

    HOSPITALS: ClassVar[list[dict]] = [
        {"id": "HOSP_001", "name": "Szpital Uniwersytecki w Krakowie", "city": "Kraków"},
        {"id": "HOSP_002", "name": "Szpital Kliniczny w Warszawie", "city": "Warszawa"},
        {"id": "HOSP_003", "name": "Centrum Kardiologiczne w Poznaniu", "city": "Poznań"},
        {"id": "HOSP_004", "name": "Szpital Specjalistyczny w Gdańsku", "city": "Gdańsk"},
        {"id": "HOSP_005", "name": "Klinika Kardiologiczna we Wrocławiu", "city": "Wrocław"},
        {"id": "HOSP_006", "name": "Szpital Miejski w Łodzi", "city": "Łódź"},
        {"id": "HOSP_007", "name": "Centrum Medyczne Katowice", "city": "Katowice"},
        {"id": "HOSP_008", "name": "Szpital Regionalny w Lublinie", "city": "Lublin"},
    ]

    def __init__(
        self,
        bootstrap_servers: str = "localhost:29092",
        topic: str = "ecg-pending",
        data_path: str = "data/ptb-xl-1.0.3",
        num_threads: int = 4,
        interval_sec: int = 5.0,
        sampling_rate: int = 100,
    ) -> None:
        """Initialize ECGProducer.

        Args:
            bootstrap_servers (str): Kafka broker addresses. Defaults to "localhost:29092".
            topic (str): Kafka topic name. Defaults to "ecg-pending".
            data_path (str): Path to PTB-XL data. Defaults to "data/ptb-xl-1.0.3".
            num_threads (int): Number of threads to simulate hospitals. Defaults to 4.
            interval_sec (float): Interval in seconds between ECG records. Defaults to 5.0.
            sampling_rate (int): Sampling rate in Hz. Defaults to 100.

        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.num_threads = num_threads
        self.interval_sec = interval_sec
        self.sampling_rate = sampling_rate

        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",  # Wait for all replicas to acknowledge the message
            retries=3,
            max_in_flight_requests_per_connection=1,  # Ensure message order per partition
            compression_type="gzip",  # Compress large ECG payloads
        )

        logger.info("Loading data from %s...", data_path)
        self.dataset = ECGDataset(
            data_path=data_path,
            sampling_rate=sampling_rate,
            split="test",
        )
        logger.info("Loaded %s ECG records.", len(self.dataset.records))

        self.stats: dict = {"sent": 0, "errors": 0, "start_time": None}
        self.stats_lock = threading.Lock()
        self.running = False

    def _create_message(self, thread_id: int) -> dict:
        """Build a Kafka message from randomly selected PTB-XL record.

        Args:
            thread_id (int): Index of the calling producer thread.

        Returns:
            dict: Dictionary ready for JSON serialization and Kafka publishing.

        """
        sample = self.dataset.get_sample_for_streaming()
        hospital = random.choice(self.HOSPITALS)
        exam_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()  # noqa: DTZ005 - No timezone needed

        return {
            "exam_id": exam_id,
            "timestamp_sent": timestamp,
            "hospital": hospital,
            "thread_id": thread_id,
            "patient": {
                "ecg_id": sample["ecg_id"],
                "age": sample["age"],
                "sex": ("M" if sample["sex"] == 0 else "F" if sample["sex"] == 1 else None),
            },
            "signal": {
                "data": sample["signal"],
                "sampling_rate": self.sampling_rate,
                "num_channels": 12,
                "duration_sec": 10.0,
                "leads": ECG_LEAD_NAMES,
            },
            "metadata": {
                "ground_truth_label": sample["label"],
                "ground_truth_name": sample["label_name"],
            },
        }

    def _producer_thread(self, thread_id: int) -> None:
        """Worker loop for a single hospital thread.

        Sends records at `interval_sec` intervals.

        Args:
            thread_id (int): Index of the producer thread.

        """
        logger.info("[Thread %s] Started", thread_id)

        while self.running:
            try:
                message = self._create_message(thread_id)

                future = self.producer.send(
                    topic=self.topic,
                    key=message["exam_id"],
                    value=message,
                )
                record_metadata = future.get(timeout=10)

                with self.stats_lock:
                    self.stats["sent"] += 1

                logger.info(
                    "[Thread %s] Sent: %s... -> partition %s | Hospital: %s | ECG ID: %s",
                    thread_id,
                    message["exam_id"][:8],
                    record_metadata.partition,
                    message["hospital"]["id"],
                    message["patient"]["ecg_id"],
                )

            except KafkaError:
                with self.stats_lock:
                    self.stats["errors"] += 1
                logger.exception("[Thread %s] Kafka error", thread_id)

            except Exception:
                with self.stats_lock:
                    self.stats["errors"] += 1
                logger.exception("[Thread %s] Unexpected error", thread_id)

            time.sleep(max(0.1, self.interval_sec + random.uniform(-0.5, 0.5)))

        logger.info("[Thread %s] Stopped", thread_id)

    def start(self) -> None:
        """Start all producer threads and block until interrupted."""
        logger.info("\n%s", "=" * 60)
        logger.info("ECG Producer - simulating hospital ECG streams")
        logger.info("\n%s", "=" * 60)
        logger.info("Kafka:    %s", self.bootstrap_servers)
        logger.info("Topic:    %s", self.topic)
        logger.info("Threads:  %s", self.num_threads)
        logger.info("Interval: %s sec", self.interval_sec)
        logger.info("\n%s", "=" * 60)

        self.running = True
        self.stats["start_time"] = time.time()

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            try:
                for i in range(self.num_threads):
                    executor.submit(self._producer_thread, thread_id=i)

                while self.running:
                    time.sleep(10)
                    self._print_stats()

            except KeyboardInterrupt:
                logger.info("\nShutdown signal received...")
                self.running = False

        self._print_stats()
        self.producer.close()
        logger.info("ECG Producer stopped.")

    def _print_stats(self) -> None:
        """Print a summary of messages sent and errors."""
        with self.stats_lock:
            elapsed = time.time() - self.stats["start_time"]
            rate = self.stats["sent"] / elapsed if elapsed > 0 else 0

            logger.info("\n--- Stats (%.0f s) ---", elapsed)
            logger.info("Sent:     %s", self.stats["sent"])
            logger.info("Errors:   %s", self.stats["errors"])
            logger.info("Rate:     %.2f msg/sec", rate)
            logger.info("%s\n", "=" * 30)

    def stop(self) -> None:
        """Signal all producer threads to stop after the current send."""
        self.running = False


def main() -> None:
    """Entry point for the ECG Kafka Producer."""
    setup_logging("logs", "producer")
    parser = argparse.ArgumentParser(description="ECG Kafka Producer")

    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:29092",
        help="Kafka broker addresses",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="ecg-pending",
        help="Kafka topic name",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ptb-xl-1.0.3",
        help="Path to PTB-XL data root directory",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of producer threads",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Interval in seconds",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=100,
        choices=[100, 500],
        help="ECG sampling rate in Hz",
    )

    args = parser.parse_args()

    bootstrap_servers = args.bootstrap_servers
    topic = args.topic
    data_path = args.data_path
    num_threads = args.num_threads
    interval_sec = args.interval
    sampling_rate = args.sampling_rate

    if not Path(data_path).exists():
        logger.error("Data path does not exist: %s", data_path)
        return

    producer = ECGProducer(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        data_path=data_path,
        num_threads=num_threads,
        interval_sec=interval_sec,
        sampling_rate=sampling_rate,
    )

    try:
        producer.start()
    except KeyboardInterrupt:
        producer.stop()


if __name__ == "__main__":
    main()
