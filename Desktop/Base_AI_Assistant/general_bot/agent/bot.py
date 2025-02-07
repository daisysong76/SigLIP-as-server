# agent/bot.py
import asyncio
from agent.meta_parser import parse_meta_input
from agent.plan_agent import PlanAgent
from agent.action_agent import ActionAgent
from agent.critic_agent import CriticAgent

class Bot:
    def __init__(self):
        self.plan_agent = PlanAgent()
        self.action_agent = ActionAgent()
        self.critic_agent = CriticAgent()

    async def run(self, meta_input: str) -> str:
        """
        Orchestrate the process:
          1. Parse the raw meta input.
          2. Generate a plan based on the parsed instruction.
          3. Execute the plan.
          4. Evaluate the results.
        """
        # Step 1: Parse meta input into structured instructions.
        instructions = parse_meta_input(meta_input)
        
        # Step 2: Generate a plan from the instructions.
        plan = await self.plan_agent.generate_plan(instructions)
        
        # Step 3: Execute the generated plan.
        results = await self.action_agent.execute_plan(plan)
        
        # Step 4: Evaluate the execution results.
        evaluation = self.critic_agent.evaluate(results)
        return evaluation

if __name__ == '__main__':
    import sys
    # Use command-line argument if provided; otherwise, default input.
    meta_input = sys.argv[1] if len(sys.argv) > 1 else "Schedule a meeting and call John Doe."
    bot = Bot()
    result = asyncio.run(bot.run(meta_input))
    print("Final Evaluation:")
    print(result)
