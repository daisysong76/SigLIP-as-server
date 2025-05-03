1. we need to start Qdrant as well since our memory system depends on both Redis and Qdrant.

2. install redis-py with async support. The aioredis package has been merged into redis package since version 4.2.0:


Redis is an open source, in-memory data structure server that functions as a high-performance NoSQL key-value store. It is designed for speed, storing all data in RAM, which enables extremely fast read and write operations compared to traditional disk-based databases. Redis supports a variety of data structures, including strings, hashes, lists, sets, sorted sets, bitmaps, hyperloglogs, geospatial indexes, and streams.
Key features of Redis include:
	•	In-memory storage: All data is kept in memory for rapid access, with optional persistence to disk for durability.
	•	Rich data types: Redis natively supports multiple data types, allowing for complex operations beyond simple key-value storage.
	•	Atomic operations: All commands are atomic, ensuring data consistency even with concurrent access.
	•	High availability and scalability: Redis offers built-in replication, automatic partitioning (Redis Cluster), and high availability via Redis Sentinel.
	•	Versatile use cases: Commonly used as a cache, database, and message broker for real-time applications, session management, leaderboards, and more.
	•	Extensibility: Redis can be extended with modules and integrates with many programming languages.
Because Redis keeps data in memory, it is best suited for scenarios where speed is critical and the dataset fits within available RAM. It can persist data to disk for recovery, but its primary advantage is ultra-fast data access.