api/                             # Backend API layer
│   ├── index.js                     # Main Node.js server for Twilio + WebSockets
│   ├── routes/                      # REST API and WebSocket handlers
│   │   ├── __init__.py
│   │   ├── agent_routes.py
│   │   ├── health_routes.py
│   │   └── graphql/                 # GraphQL integration (if needed)
│   │       └── schema.py
│   ├── middleware/                  # Middleware (auth, logging, etc.)
│   │   ├── __init__.py
│   │   └── logging.py
│   ├── grpc/                        # gRPC services
│   │   └── services/
│   └── websockets/                  # WebSocket handlers for real-time data
│       └── handlers.py