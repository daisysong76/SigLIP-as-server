# Multi-Agent Bot Project

A modular, asynchronous bot designed to perform various business-critical tasks by integrating multiple external tools and services. This bot uses a collection of sub-agents to parse commands, plan tasks, execute tool calls (e.g., calendar events, phone calls, emails, browsing, contact search, and search), and then evaluate the outcomes.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project implements a multi-agent bot that can be used to manage various operational tasks such as:
- **Browsing**: Simulate web browsing or information gathering.
- **Calendar**: Schedule events and manage calendars.
- **Call**: Initiate phone calls via integrated telephony tools.
- **Contact Search**: Look up and manage contact data.
- **Email**: Send emails using external email APIs.
- **Search**: Perform searches for data or content.

The bot is built in a modular way with distinct agents responsible for:
- **Meta Parsing:** Converting raw input into structured instructions.
- **Task Planning:** Generating a detailed plan for the requested tasks.
- **Action Execution:** Calling external tools (integrations) to perform the tasks.
- **Critiquing:** Evaluating and summarizing the results of task execution.

## Features

- **Modular Architecture:** Each function (parsing, planning, executing, evaluating) is handled by its own component.
- **Tool Integration:** Easily extendable to integrate with real-world services (Google Calendar, Twilio, Email APIs, etc.).
- **Asynchronous Execution:** Utilizes Python’s `asyncio` to efficiently handle I/O-bound operations.
- **Command-Line Interface:** Run and test the bot directly from the command line.
- **Extensibility:** New tasks and integrations can be added with minimal changes to the core logic.

## Directory Structure

```plaintext
project/
├── main.py
├── pyproject.toml
├── README.md
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── .editorconfig
├── agent/
│   ├── __init__.py
│   ├── bot.py                # Main orchestrator that ties all agents together
│   ├── meta_parser.py        # Parses raw meta input into structured instructions
│   ├── plan_agent.py         # Generates a plan (list of tasks) from instructions
│   ├── action_agent.py       # Executes each task by calling external tools
│   └── critic_agent.py       # Evaluates the results of task execution
├── integrations/
│   ├── __init__.py
│   ├── google_calendar_integration.py   # Simulated integration for calendar tasks
│   ├── twilio_integration.py              # Simulated integration for call tasks
│   └── email_integration.py               # Simulated integration for email tasks
├── tests/                                  # Unit, integration, and e2e tests
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── ...                                     # Additional directories (docs, scripts, etc.)
