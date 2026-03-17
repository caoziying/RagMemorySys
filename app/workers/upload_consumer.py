"""上传队列消费者。"""

from __future__ import annotations

import asyncio
from datetime import datetime

from app.core.config import settings
from app.core.logger import get_logger, setup_logging
from app.memory.manager import MemoryManager
from app.memory.profile import ProfileManager
from app.retrieval.retriever import Retriever
from app.workers.upload_queue import UploadQueue

setup_logging()
logger = get_logger(__name__)


class UploadConsumer:
    """持续消费 upload 队列消息。"""

    def __init__(self, poll_interval: float | None = None, retry_limit: int | None = None) -> None:
        self.poll_interval = poll_interval or settings.upload_worker_poll_interval
        self.queue = UploadQueue(retry_limit=retry_limit or settings.upload_retry_limit)
        self.retriever = Retriever()
        self.memory_manager = MemoryManager()
        self.profile_manager = ProfileManager()

    async def run_forever(self) -> None:
        logger.info("Upload consumer started.")
        while True:
            message = await self.queue.get_next_message()
            if message is None:
                await asyncio.sleep(self.poll_interval)
                continue

            request_id = message["request_id"]
            try:
                if await self.queue.is_processed(request_id):
                    logger.warning("检测到重复 request_id，直接 ack | request_id={}", request_id)
                    await self.queue.ack(message["id"])
                    continue

                payload = message["payload"]
                user_id = payload["user_id"]
                texts = payload["texts"]
                timestamp = datetime.fromisoformat(payload["timestamp"])

                chunks_stored = await self.retriever.store(
                    user_id=user_id,
                    texts=texts,
                    timestamp=timestamp,
                )
                await self.memory_manager.update(user_id=user_id, new_texts=texts)
                await self.profile_manager.extract_and_update_profile(user_id, "\n".join(texts))

                await self.queue.mark_processed(request_id)
                await self.queue.ack(message["id"])
                logger.info(
                    "Upload 消费成功 | request_id={} | user_id={} | chunks_stored={}",
                    request_id,
                    user_id,
                    chunks_stored,
                )
            except Exception as exc:
                logger.exception("Upload 消费失败，将重试 | request_id={} | error={}", request_id, exc)
                await self.queue.fail(message, str(exc))


async def main() -> None:
    consumer = UploadConsumer()
    await consumer.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
