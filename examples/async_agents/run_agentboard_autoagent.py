# -*- coding: utf-8 -*-
# @Time    : 2024/06/27

import time
import agentboard as ab
from agentboard.utils import function_to_schema

from AutoAgent.core import AsyncAgent, AsyncAutoEnv
from AutoAgent.utils import get_current_datetime

class AutoAgentWorkflow(AsyncAgent):
    """
        Agent who can plan a day and write blog/posts videos on the website
        execute cycle: plan->act->reflect
    """
    async def run_loop(self):

        try:

            ab.summary.agent_loop(name="START", data="This is start stage of %s" % self.name, agent_name = self.name, process_id="START", duration = 0)
            print ("%s|%s|START|%s" % (self.name, get_current_datetime(), 0))

            ## Plan stage
            # input
            ab.summary.agent_loop(name="INPUT", data="This is Plan Input of %s" % self.name, agent_name = self.name, process_id="PLAN", duration = 0)

            plan_duration = await self.plan()

            ab.summary.agent_loop(name="EXECUTION", data="This is Execution stage of %s" % self.name, agent_name = self.name, process_id="PLAN", duration = plan_duration)
            ab.summary.agent_loop(name="OUTPUT", data="This is Plan Output of %s" % self.name, agent_name = self.name, process_id="PLAN", duration = 0)
            print ("%s|%s|PLAN Complete|%s" % (self.name, get_current_datetime(), plan_duration))


            ## ACT stage
            ab.summary.agent_loop(name="INPUT", data="This is Act Input of %s" % self.name, agent_name = self.name, process_id="ACT", duration = 0)

            act_duration = await self.act()

            ab.summary.agent_loop(name="EXECUTION", data="This is ACT Execution Input of %s" % self.name, agent_name = self.name, process_id="ACT", duration = act_duration)
            ab.summary.agent_loop(name="OUTPUT", data="This is ACT Output of %s" % self.name, agent_name = self.name, process_id="ACT", duration = 0)
            print ("%s|%s|ACT Complete|%s" % (self.name, get_current_datetime(), act_duration))

            reflect_duration = await self.reflect()
            ab.summary.agent_loop(name="REFLECTION", data="This is Reflection stage of %s" % self.name, process_id="REFLECT", agent_name = self.name, duration = reflect_duration)
            print ("%s|%s|REFLECTION Complete|%s" % (self.name, get_current_datetime(), reflect_duration))

            ## add a decision node log
            ab.summary.agent_loop(name="DECISION", data="This is decision stage of %s" % self.name, process_id="DECISION", agent_name = self.name, workflow_type = "decision", duration = 0)

            total_duration = plan_duration + act_duration + reflect_duration
            ab.summary.agent_loop(name="END", data="This is end stage of %s" % self.name, process_id="END", agent_name = self.name, duration = total_duration)
            print ("%s|%s|END|%s" % (self.name, get_current_datetime(), 0))

            return "%s|%s|run_loop complete|%s" % (self.name, get_current_datetime(), total_duration)
        except Exception as e:
            print (e)
            return "%s|%s|run_loop complete|%s" % (self.name, get_current_datetime(), 0)

def get_openai_client():
    from openai import OpenAI
    # api_key = os.environ.get("OPENAI_API_KEY")
    api_key = "xxxxxxxx"
    client = OpenAI(api_key=api_key)
    return client

def run_async_agents_env():
    """
    """
    ## Website Content Auditor Agent (Decide if contents are suitable to publish or if the contents are spam)
    agent1_prompt = """You are playing the role of a web admin agent... """
    ## Website Automatic Reply Agent
    agent2_prompt = """You are playing the role of a automatic reply agent..."""
    ## User Publish New Content -> Make Newly Published Content status from pending audit to online  -> comment reply bot will reply to new comment.
    agent_1 = AsyncAgent(name="agent 1", instructions=agent1_prompt)
    agent_2 = AsyncAgent(name="agent 2", instructions=agent2_prompt)

    agents = [agent_1, agent_2]
    env = AsyncAutoEnv(get_openai_client(), agents=agents)
    results = env.run()

def run_async_agents_env_agentboard():
    """
    """
    with ab.summary.FileWriter(logdir="./log", static="./static") as writer:
        agent1_prompt = """You are playing the role of a Web admin agent... """
        agent2_prompt = """You are playing the role of a Automatic reply agent..."""

        agent_1 = AutoAgentWorkflow(name="agent 1", instructions=agent1_prompt)
        agent_2 = AutoAgentWorkflow(name="agent 2", instructions=agent2_prompt)

        agents = [agent_1, agent_2]
        env = AsyncAutoEnv(get_openai_client(), agents=agents)
        results = env.run()

if __name__ == "__main__":
    run_async_agents_env_agentboard()
