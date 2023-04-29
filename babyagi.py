#!/usr/bin/env python3
import os
import subprocess
import time
import re
from collections import deque
from typing import Dict, List

import openai
import pinecone
from dotenv import load_dotenv

from datastore import DataStore, get_ada_embedding
from tree import Node, unpack_tree
from utils import save_checkpoint, maybe_load_checkpoint

# Load default environment variables (.env)
load_dotenv()

# Engine configuration

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

if "gpt-4" in OPENAI_API_MODEL.lower():
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
assert (
    PINECONE_ENVIRONMENT
), "PINECONE_ENVIRONMENT environment variable is missing from .env"

# Table config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Goal configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))

DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
    os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)
if ENABLE_COMMAND_LINE_ARGS:
    from extensions.argparseext import parse_arguments

    OBJECTIVE, INITIAL_TASK, OPENAI_API_MODEL, DOTENV_EXTENSIONS = parse_arguments()

VERBOSE = (os.getenv("VERBOSE", "false").lower() == "true")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "")
STOP_CRITERION = os.getenv("STOP_CRITERION", "")
SKIP_STOP_CRITERION = (os.getenv("SKIP_STOP_CRITERION", "false").lower() == "true")

# Load additional environment variables for enabled extensions
if DOTENV_EXTENSIONS:
    from extensions.dotenvext import load_dotenv_extensions

    load_dotenv_extensions(DOTENV_EXTENSIONS)

# TODO: There's still work to be done here to enable people to get
# defaults from dotenv extensions # but also provide command line
# arguments to override them

if "gpt-4" in OPENAI_API_MODEL.lower():
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

# Print OBJECTIVE
print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")

print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")

# Configure OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY

if PINECONE_API_KEY:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    # Create Pinecone index
    table_name = YOUR_TABLE_NAME
    dimension = 1536
    metric = "cosine"
    pod_type = "p1"
    if table_name not in pinecone.list_indexes():
        pinecone.create_index(
            table_name, dimension=dimension, metric=metric, pod_type=pod_type
        )

    # Connect to the index
    index = pinecone.Index(table_name)
else:
    print("PINECONE_API_KEY environment variable is missing from .env, using local datastore instead")
    index = DataStore()

# Task list
task_list = deque([])
task_id_counter = 1


def add_tasks(tasks: List[Node]):
    for task in tasks:
        task_list.append(task)


def parse_tasks(result):
    global task_id_counter
    new_tasks = result.split("\n")
    tasks = []
    for task_name in new_tasks:
        match = re.match(r"^\d+\.", task_name) # enumerated task
        if match:
            task_name = task_name.strip().split(".", 1)
            task_name = task_name[1].strip()
        if len(task_name) > 20: # must be more than 20 chars
            task_id_counter += 1
            task_node = Node(task_id=task_id_counter, task_name=task_name, state="new")
            tasks.append(task_node)
    return tasks


def openai_call(
    prompt: str,
    model: str = OPENAI_API_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 100,
):
    while True:
        try:
            if model.startswith("llama"):
                # Spawn a subprocess to run llama.cpp
                cmd = cmd = ["llama/main", "-p", prompt]
                result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                return result.stdout.strip()
            elif not model.startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                # Use chat completion API
                messages = [{"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


def task_creation_agent(task: Node):
    tasks_to_do = [t.task_name for t in task_list]
    prompt = f"""
    You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {OBJECTIVE},
    The last completed task has the result: {task.result}.
    This result was based on this task description: {task.task_name}. These are incomplete tasks: {', '.join(tasks_to_do)}.
    Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
    Return the tasks as a numbered list.\n"""
    response = openai_call(prompt)
    if VERBOSE:
        print(prompt, response)
    tasks = parse_tasks(response)
    return tasks


def prioritization_agent():
    global task_list
    task_names = [f"{i+1}. {t.task_name}" for i, t in enumerate(task_list)]
    task_names = "\n".join(task_names)
    prompt = f"""
    You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
    Consider the ultimate objective of your team:{OBJECTIVE}.
    Remove tasks if they are low impact. Return the new order of indicies separated by commas where the first is the highest priority.\n"""
    response = openai_call(prompt)
    if VERBOSE:
        print(prompt, response)
    reorder = [int(x) for x in response.strip().split(", ")]
    
    to_remove = list(set(range(1,len(task_list)+1))-set(reorder))
    for i in to_remove:
        task = task_list[i-1]
        task.state = "cancelled"

    task_names_reordered = []
    for i in reorder:
        task = task_list[i-1]
        task.state = "queued"
        task_names_reordered.append(task)
    task_list = deque(task_names_reordered)


def execution_agent(objective: str, task: str) -> str:
    context = context_agent(query=objective, n=5)
    # print("\n*******RELEVANT CONTEXT******\n")
    # print(context)
    prompt = f"""
    You are an AI who performs one task based on the following objective: {objective}.\n
    Take into account these previously completed tasks: {context}.\n
    Your task: {task}\nResponse:"""
    response = openai_call(prompt, temperature=0.7, max_tokens=2000)
    if VERBOSE:
        print(prompt, response)
    return response


def context_agent(query: str, n: int):
    if isinstance(index, DataStore):
        context, _ = index.query(query, top_k=n)
    else:
        query_embedding = get_ada_embedding(query)
        results = index.query(query_embedding, top_k=n, include_metadata=True, namespace=OBJECTIVE)
        # print("***** RESULTS *****")
        # print(results)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        context = [(str(item.metadata["task"])) for item in sorted_results]
    return context


def stop_criterion_agent(task):
    if SKIP_STOP_CRITERION:
        return False
    prompt = f"""
    You are a stop criterion AI that decides if a task should be broken down further or not based on the task result.
    Task: {task.task_name}
    Result:\n{task.result}
    Return "Yes" if the STOP_CRITERION is met and "No" if it is not. You only need to respond with a single word.
    STOP_CRITERION: {STOP_CRITERION}
    Decision:"""
    response = openai_call(prompt)
    if VERBOSE:
        print(prompt, response)
    if "no" in response.lower():
        return False
    elif "yes" in response.lower():
        return True
    else:
        print(
            f"Warning: stop criterion returned {response}, assuming it shouldn't be stopped..."
        )
        return False


objective_node, embedded_data = maybe_load_checkpoint(CHECKPOINT_DIR)
if embedded_data:
    index.load_data(embedded_data)

if not objective_node:
    # Objective task
    objective_node = Node(task_id=0, task_name=OBJECTIVE, state="complete")

    # Add the first task
    first_task = Node(task_id=1, task_name=INITIAL_TASK, state="queued")
    objective_node.children = [first_task]

    first_task.result = execution_agent(OBJECTIVE, first_task.task_name)
    first_task.state = "complete"
    print("\033[93m\033[1m" + "\n*****INITIAL TASKS*****\n" + "\033[0m\033[0m")
    print(first_task.result)
    tasks = parse_tasks(first_task.result)
    add_tasks(tasks)
    first_task.children = tasks
else:
    tasks = unpack_tree(objective_node)
    task_list = deque([task for task in tasks if task.state == "queued"])
    for task in tasks:
        if task.task_id > task_id_counter:
            task_id_counter = task.task_id

# Main loop
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in task_list:
            print(str(t.task_id) + ": " + t.task_name)

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task.task_id) + ": " + task.task_name)

        # Send to execution function to complete the task based on the context
        task.result = execution_agent(OBJECTIVE, task.task_name)
        task.state = "complete"
        
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(task.result)

        # Step 2: Store result in vector database
        if isinstance(index, DataStore):
            index.upsert(task.task_id, task.task_name, task.result)
        else:
            vector = get_ada_embedding(task.result)
            index.upsert(
                [(task.task_id, vector, {"task": task.task_name, "result": task.result})],
            namespace=OBJECTIVE
            )
        
        stop_status = stop_criterion_agent(task)
        print("\033[93m\033[1m" + "\n*****STOP STATUS*****\n" + "\033[0m\033[0m")
        print(stop_status)

        if not stop_status:
            # Step 3: Create new tasks and reprioritize task list
            tasks = task_creation_agent(task)
            add_tasks(tasks)
            task.children = tasks

            prioritization_agent()

        save_checkpoint(CHECKPOINT_DIR, objective_node, index._data)
    else:
        break

    time.sleep(1)  # Sleep before checking the task list again
