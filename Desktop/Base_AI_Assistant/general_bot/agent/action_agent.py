# agent/action_agent.py
import asyncio

class ActionAgent:
    async def execute_plan(self, plan: list) -> list:
        """
        Execute each task in the plan sequentially and return the list of results.
        """
        results = []
        for task in plan:
            result = await self.execute_task(task)
            results.append(result)
        return results

    async def execute_task(self, task: dict) -> str:
        """
        Execute a single task based on its type.
        Here we simulate execution by returning a message that includes
        the task details (title, commit message, and commit date).
        """
        await asyncio.sleep(0.5)  # Simulate action delay
        
        task_type = task.get("task")
        details = task.get("details", {})
        title = details.get("title", "Unknown Task")
        commit_message = details.get("last_commit_message", "N/A")
        commit_date = details.get("last_commit_date", "N/A")

        # Simulate task execution by returning a formatted string.
        return (f"[{title}] Executed with commit message '{commit_message}' "
                f"(Committed: {commit_date}).")
