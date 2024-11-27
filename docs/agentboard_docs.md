# AgentBoard: AI Agent Visualization Toolkit Board Document

DeepNLP AgentBoard is a visualization toolkit (similar to Tensorboard to visualize tensors) to visualize and monitor the agent loops and key entities of AI Agents development, such as messages, tools/functions, workflow and raw data types including text, dict or json, image, audio, video, etc. You can easily add logs with agentboard together with various AI agent frameworks, such as AutoGen, Langgraph, AutoAgent. 

You can install and import the 'agentboard' python package and use functions under a 'with' block. See quickstart for install and run agentboard [See full details of Quickstart](docs/quickstart.md).

![agentboard tool function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_loop_workflow_hint.jpg?raw=true)


## AgentBoard Supported AI Agent Loop Elements and Data Types

|  Functions  | DataType |  Description  |
|  -------- | --------  | --------  |
|  [**ab.summary.messages**](#absummarymessages) | message |   List of messages, json format [{"role": "user", "content": "content_1"}, {"role": "assistant", "content": "content_2"}] |
|  [**ab.summary.tool**](#absummarytool) |  function |   User defined functions, The schema of the functions which are passed to LLM API calling, Support OpenAI and Anthropic stype  |
|  [**ab.summary.agent_loop**](#absummaryagent_loop) |  str |   User defined Agent Loop with various stages PLAN/ACT/REFLECT/etc  |
|  [**ab.summary.RAGPipeline**](#absummaryragpipeline) |  str |  RAGPipeline is a class to wrap the RAG input(query, query embedding) and RAG output (docs, relevance score), etc and display the workflow in agentboard |
|  [**ab.summary.text**](#absummarytext)  |  str |   Text data, such as prompt, assistant responded text  |
|  [**ab.summary.dict**](#absummarydict)  |  dict  |   Dict data, such as input request, output response, class __dict__ |
|  [**ab.summary.image**](#absummaryimage)  | tensor |   Support both torch.Tensor and tf.Tensor,  torch.Tensor takes input shape [N, C, H, W], N: Batch Size, C: Channels, H: Height, W: Width; tf.Tensor, input shape [N, H, W, C], N: Batch Size, H: Height, W: Width, C: Channels.  |
|  [**ab.summary.audio**](#absummaryaudio)   | tensor |  Support torch.Tensor data type. The input tensor shape [B, C, N], B for batch size, C for channel, N for samples. |
|  [**ab.summary.video**](#absummaryvideo)  | tensor |  Support torch.Tensor data type. The input tensor shape should match [T, H, W, C], T: Number of frames, H: Height, W: Width, C: Number of channels (usually 3 for RGB) |



## AgentBoard Supported Web Interface GUI for AI Agents Simulation, Autonomous Planning

![agentboard summary messages function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/auto_plan_agent_1.jpg?raw=true)


|  Web Inteface | AI Agent Roles |Description  |
|  -------- | --------  | --------  |
|  **X(twitter) Style Social Media** | [Normal Users](#agent-normal-users) | AI Agents performs the role of normal users, who can interact with the GUI, e.g. Post on X (twitter), Comment, Reply, Like, Follow, etc on the website interface through APIs|
|  **X(twitter) Style Social Media** | [Website Admin](#agent-website-admin) | AI Agents performs the role as the website admin and audit post content to see if it violates community standards |
|  **X(twitter) Style Social Media** | [Website Automatic Comment Bot](#agent-website-automatic-comment-bot) | AI Agents perform the role as automatic replying robot to posts that other AI Agents publish|



### ab.summary.messages
- **Description**: Write LLM chat history as list of message with keys 'role' and 'content' to logs and display in a web chatbot for visualization. The data format of messages is 
like [{"role":"user", "content": "user content 1"}, {"role":"assistant", "content": "assistant reply 1"}, 
  {"role":"user", "content": "user content 2"}].


```   
ab.summary.messages(
    name: str, 
    data: list, 
    agent_name = None,
    process_id = None,
    **kwargs
)
```

- **Arguments**:
  - `name` (str): Name of the chat history of messages.
  - `data` (list): List of python dict, data format messages in OpenAI and other Chat History format. 
  - `agent_name` (str): Name of Agent, agent_name is the unique id to identify the workflows of each individual agent in the multi-agent environment.
  - `process_id` (str): The defined process id of agent loop, such as PLAN, ACT, REFLECT, REACT. The process id will be used to group workflow shapes of the agent loop, such as grouping all the workflows in the "PLAN" stage. 


- **Returns**:
  The messages will write one line of log in json dict format with below keys: 
  - `name` (str): Name of the chat history of messages.
  - `data` (str): Json Str of chat messages history.
  - `data_type` (str): message
  - `timestamp` (str): timestamp of running the function.
  - `process_id` (str): process_id is used to group by several workflow_id, such as in PLAN stage, Act stage, etc.
  - `workflow_id` (str): an unique id of UUID4() indicating one step in the agent loop workflow.
  - `workflow_type` (str): Optional. 
  - `agent_name` (str): Optional. agent_name is used to identify by workflows belonging to which agent.


![agentboard summary messages function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_chat_visualizer.jpg?raw=true)

[See full details of ab.summary.messages](docs/summary_messages.md)


**Logs format for ab.summary.messages**

```
{"data": "[{\"role\": \"user\", \"content\": \"hello\"}, {\"role\": \"assistant\", \"content\": \"Hola! My name is bot.\"}, {\"role\": \"user\", \"content\": \"Please help me summarize the stock market news.\"}]", "name": "Chatbot Messages", "data_type": "messages", "timestamp": 1732092955096, "workflow_id": "1df86095-85f3-4287-b3e1-ba26fd666524", "process_id": "chat", "agent_name": "chatbot"}
```


### ab.summary.tool
- **Description**: Write Tool Function calling of LLM related information to logs, such as the schema of functions, function input, function output, etc.


```
    
ab.summary.tool(
    name: str, 
    data: list, 
    agent_name = None,
    process_id = None,
    **kwargs
)


```

- **Arguments**:
  - `name` (str): Name of the tool/function calling used to log.
  - `data` (list): list of original python functions as tools, such as get_weather(), calculate(), the functions are converted to OpenAI ChatGPT/Anthropic Claude functions schema json format and passed to LLM API Callings. The function schema will be logged.
  - `agent_name` (str): Name of Agent, agent_name is the unique id to identify the workflows of each individual agent in the multi-agent environment.
  - `process_id` (str): The defined process id of agent loop, such as PLAN, ACT, REFLECT, REACT. The process id will be used to group workflow shapes of the agent loop, such as grouping all the workflows in the "PLAN" stage. 


- **Returns**:
  The messages will write one line of log in json dict format with below keys: 
  - `name` (str): Name of the tool/function calling.
  - `data` (str): Json Str of tool schema with arguments type definition.
  - `data_type` (str): tool.
  - `timestamp` (str): timestamp of running the function.
  - `workflow_id` (str): an unique id of UUID4() indicating one step in the agent loop workflow.
  - `agent_name` (str): agent_name is used to identify by workflows belonging to which agent.
  - `process_id` (str): process_id is used to group by several workflow_id, such as in PLAN stage, Act stage, etc.


![agentboard tool function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_tool.jpg?raw=true)


[See full details of ab.summary.tool](docs/summary_tool.md)

**Logs format for ab.summary.tool**

```
{"data": "{\"type\": \"function\", \"function\": {\"name\": \"calling_bing_tools\", \"description\": \"\", \"parameters\": {\"type\": \"object\", \"properties\": {\"keyword\": {\"type\": \"string\"}, \"limit\": {\"type\": \"integer\"}}, \"required\": [\"keyword\", \"limit\"]}}}", "name": "calling_bing_tools", "data_type": "tool", "timestamp": 1732092955096, "workflow_id": "af2888b4-18b9-4238-8b8b-c04707762c61", "process_id": "act", "agent_name": "agent 2"}
```


### ab.summary.agent_loop
- **Description**: Write summary of user defined Agent Loop with input, output and data of various stages PLAN/ACT/REFLECT/etc, displayed in the worflow chart on agentboard.

```
    
ab.summary.agent_loop(
    name: str, 
    data: dict/str,
    agent_name = None,
    process_id = None,
    workflow_type = None,
    **kwargs
)


```

- **Arguments**:
  - `name` (str): Name of the worflow log step, such as "Input of Plan", "Output of Act", etc.
  - `data` (dict/str): The log content you want to display on the agentboard, such as the parameters extracted from user query, such as {"city_name":"New York"} in a weather fetch loop.
  - `agent_name` (str): Name of Agent, agent_name is the unique id to identify the workflows of each individual agent in the multi-agent environment.
  - `process_id` (str): The defined process id of agent loop, such as PLAN, ACT, REFLECT, REACT. The process id will be used to group workflow shapes of the agent loop, such as grouping all the workflows in the "PLAN" stage. 


- **Returns**:
  The messages will write one line of log in json dict format with below keys: 
  - `name` (str): Name of the tool/function calling.
  - `data` (str): Json Str of tool schema with arguments type definition.
  - `data_type` (str): tool.
  - `timestamp` (str): timestamp of running the function.
  - `workflow_id` (str): an unique id of UUID4() indicating one step in the agent loop workflow.
  - `workflow_type` (str): Support various workflow_type enum to display on agentboard, such as 'start', 'end', 'process', 'decision', 'rag', 
  - `agent_name` (str): agent_name is used to identify by workflows belonging to which agent.
  - `process_id` (str): process_id is used to group by several workflow_id, such as in PLAN stage, Act stage, etc. The logs with the same process_id will become a group, such as "PLAN Input", "Plan Output", "Plan Execution".


**Log Format for ab.summary.agent_loop**

```

{"name": "INPUT", "data": "This is Plan Input of agent 1", "agent_name": "agent 1", "process_id": "PLAN", "data_type": "agent_loop", "timestamp": 1732158453846, "workflow_id": "defe2460-85aa-4ec4-88b9-cb7597f69f97", "workflow_type": "process", "duration": 0}
{"name": "EXECUTION", "data": "This is Execution stage of agent 1", "agent_name": "agent 1", "process_id": "PLAN", "data_type": "agent_loop", "timestamp": 1732158458847, "workflow_id": "62dba117-31b2-45e4-b576-525bebff48b0", "workflow_type": "process", "duration": 5}
{"name": "OUTPUT", "data": "This is Plan Output of agent 1", "agent_name": "agent 1", "process_id": "PLAN", "data_type": "agent_loop", "timestamp": 1732158458848, "workflow_id": "a1c98c01-027d-405b-b1ea-6334330f7da0", "workflow_type": "process", "duration": 0}
{"name": "DECISION", "data": "This is decision stage of agent 1", "agent_name": "agent 1", "process_id": "DECISION", "data_type": "agent_loop", "timestamp": 1732158472849, "workflow_id": "bd650777-24db-4c04-b679-8cb8f009e169", "workflow_type": "decision", "duration": 0}

```

![agentboard tool function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_loop_workflow_hint.jpg?raw=true)
[See full details of ab.summary.tool](docs/summary_agent_loop.md)



### ab.summary.RAGPipeline
- **Description**: Write summary to logs of a RAGPipeline, including various search types, such as vector/embedding search, keywords search, API calling, etc.

```
    
class RAGPipeline(BasePipeline):

      def input(self, query, embedding, **kwargs):

      def output(self, docs, scores, key_doc_id="doc_id", key_content="content", key_doc_embedding="embedding", **kwargs):

      def write(self):

```

- **ab.summary.RAGPipeline.input arguments**:
  - `query` (list of str): The list of raw query from batched user input, such as ["What's the defination of RAG?", "What's the stock price of major US market?"], etc.
  - `embedding` (2D list of float): The query embedding represented as 2D list of floats, e.g. [[0.1, 0.2, 0.3, 0.4], [ 0.5, 0.6, 0.7, 0.8]], etc. The length of 1st dimension of embeddings should equal to the length of query list.
  - `kwargs` (dict): other information as dict you want to log as the input of RAG, such as research type, etc.

- **ab.summary.RAGPipeline.output arguments**:
  - `docs` (2D list of dict): docs represent retrieved output as 2D list of dict, such as [{"doc_id": 10, "content": "content for doc 10", "embedding": [0.1, 0.2, ...]}, {"doc_id": 3, "content": "content for doc 3", "embedding": [0.1, 0.2, ...] }, ..., ] for each input query in the query list.
  - `scores` (2D list of float): The relevances score of each <query, doc> pair as embedding similarity or other metrics.
  - `key_doc_id` (str): Optional, default to "doc_id". The key should match the one in the docs list of dict object.
  - `key_content` (str): Optional, default to "content". The key should match the one in the docs list of dict object.
  - `key_doc_embedding` (str): Optional, default to "embedding". The key should match the one in the docs list of dict object.


- **Returns**:
  The messages will write one line of log in json dict format with below keys: 
  - `name` (str): Name of the RAGPipeline.
  - `data` (dict): dict of json, containing the input and output of the RAG process.
  - `data_type` (str): rag.
  - `timestamp` (str): timestamp.
  - `workflow_id` (str): an unique id of UUID4() indicating one step in the agent loop workflow.
  - `workflow_type` (str): Support various workflow_type enum to display on agentboard, such as 'start', 'end', 'process', 'decision', 'rag', 
  - `agent_name` (str): agent_name is used to identify by workflows belonging to which agent.
  - `process_id` (str): Optional.


**Log Format for ab.summary.RAGPipeline**

```
{"data":{"input":{"query":["What is the definition of RAG technology?","Does RAG requires vector databases?"],"embedding":[[0.01244938040860566,0.29050192870195146,0.35846870002353104,0.4255249256606397,0.34859658991909703,0.8259580810521346,0.8799086593834341,0.0909663223925038,0.0664290448336915,0.8684512243398872,0.24275252828207627,0.6053749349837969,0.10277184022420616,0.15160244811104762,0.9211177224557924,0.6460383677047881,0.8399952040432428,0.6177520616730009,0.24892861042698167,0.9720899330906376,0.48128032879400107,0.700580373186484,0.7317606219615412,0.9421170205659436,0.5168538365476898,0.1651233890337026,0.4525652542089922,0.24217524072371177,0.7286741032495807,0.7375463596899732,0.3865245057710115,0.747169137523689,0.10594748754631478,0.2405720811006663,0.42036826068770516,0.08394147707122535,0.7503685394003087,0.46724019757237367,0.8565872464009568,0.7650736679828678,0.9399262166387848,0.7331785238575698,0.5853776636620075,0.49816003085179994,0.14043625714192465,0.5602843552960423,0.3078267538141649,0.907247535925652,0.4853723605989222,0.21321631564225274,0.08112650360102136,0.9260944410770651,0.8633190566853475,0.6916932163057159,0.43378450211523234,0.6488168787695957,0.15798503566621724,0.07808304825782508,0.1950978974780131,0.8955630698211432,0.3251061689173006,0.8187863409928545,0.9497917830281727,0.749740587921724],[0.7731199732512269,0.0634398365622314,0.8294447786376292,0.9267618254738501,0.39184205663781935,0.9392302775123739,0.8196949882263698,0.7136196656761187,0.34910817389510607,0.12161146925686073,0.02800658197674777,0.9795581800821481,0.017778234466341192,0.24432508648969375,0.2776659001181687,0.5151640536337115,0.06822125700225623,0.3242748178778353,0.13376736332577244,0.9426327360819707,0.8457120293016857,0.9982170117321161,0.6557491325878316,0.11085289936965192,0.7029394808031871,0.29344377953654066,0.6190821284401375,0.7928363086311202,0.4132547307966292,0.8703843991143014,0.7256471616134937,0.8942386333147452,0.7105236656038937,0.8040463890107213,0.3622514275016401,0.8913921555601777,0.13351284800638363,0.4452666159363472,0.8736534967089823,0.8292159624669527,0.8039379805461747,0.29424718185306764,0.5903357571407453,0.9232952265704848,0.7672172323356062,0.24520906895491945,0.22970941091936004,0.5230318667423794,0.2537327140010852,0.8034164746596311,0.4583229734216958,0.6411997523277642,0.376901080056242,0.9425854270775265,0.4460178137037374,0.38835539783316,0.11699805167087707,0.02304115130603146,0.9623791057289419,0.6201294645297425,0.3126067347318532,0.8010286587355127,0.792766204495168,0.6487157920513426]]},"output":{"docs":[[{"doc_id":10,"content":"queries - Discusses what information you should gather along with your test queries, provides guidance on generating synthetic queries and queries that your documents don't cover.Chunking phaseUnderstand chunking economics - Discusses the factors to consider when looking at the overall cost of your chunking"},{"doc_id":7,"content":"fields created from the content in the chunks to discrete fields, such as title, summary, and keywords.Embed chunks - Uses an embedding model to vectorize the chunk and any other metadata fields that are used for vector searches.Persists chunks - Stores the chunks in the search index.RAG design and evaluation"},{"doc_id":12,"content":"the different approaches to chunking like sentence-based, fixed-size, custom, large language model augmentation, document layout analysis, using machine learning modelsUnderstand how document structure affects chunking - Discusses how the degree of structure a document has influences your choice for a"},{"doc_id":20,"content":"by running multiple experiments, persisting, and evaluating the resultsStructured approachBecause of the number of steps and variables, it's important to design your RAG solution through a structured evaluation process. Evaluate the results of each step and adapt, given your requirements. While you should"},{"doc_id":18,"content":"groundedness, completeness, utilization, and relevancyUnderstand similarity and evaluation metrics - Provides a small list of similarity and evaluation metrics you can use when evaluating your RAG solutionUnderstand importance of documentation, reporting, and aggregation - Discusses the importance of"}],[{"doc_id":7,"content":"fields created from the content in the chunks to discrete fields, such as title, summary, and keywords.Embed chunks - Uses an embedding model to vectorize the chunk and any other metadata fields that are used for vector searches.Persists chunks - Stores the chunks in the search index.RAG design and evaluation"},{"doc_id":18,"content":"groundedness, completeness, utilization, and relevancyUnderstand similarity and evaluation metrics - Provides a small list of similarity and evaluation metrics you can use when evaluating your RAG solutionUnderstand importance of documentation, reporting, and aggregation - Discusses the importance of"},{"doc_id":10,"content":"queries - Discusses what information you should gather along with your test queries, provides guidance on generating synthetic queries and queries that your documents don't cover.Chunking phaseUnderstand chunking economics - Discusses the factors to consider when looking at the overall cost of your chunking"},{"doc_id":5,"content":"the query, packages them as context within a prompt, along with the query, and sends the prompt to the large language model. The orchestrator returns the response to the intelligent application for the user to read.The following is a high-level flow for a data pipeline that supplies grounding data for"},{"doc_id":16,"content":"- Discusses some key decisions you must make for the vector search configuration that applies to vector fieldsUnderstanding search options - Provides an overview of the types of search you can consider such as vector, full text, hybrid, and manual multiple. Provides guidance on splitting a query into"}]],"score":[[7.273412460159454,7.01658178076193,6.998966861431219,6.881677896695761,6.839106694043705],[7.299657416999697,7.035988395346784,6.983690027348195,6.931392305745279,6.901445864805283]],"key_doc_id":"doc_id","key_doc_content":"content","key_doc_embedding":"embedding"}},"data_type":"rag","timestamp":1732614851455,"workflow_id":"7facc19d-f218-4768-9e78-7cb359016b78","name":"RAG 1","agent_name":"Agent RAG","process_id":"RAG","workflow_type":"rag"}

```


![agentboard tool function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_loop_workflow_hint.jpg?raw=true)
[See full details of ab.summary.tool](docs/summary_rag_pipeline.md)



### ab.summary.text
- **Description**: Write Text Data to logs, such as the prompt, user input query, response context, RAG return document plan text, etc.


```
    
ab.summary.text(
    name: str, 
    data: str, 
    agent_name = None,
    process_id = None, 
    **kwargs
)


```

- **Arguments**:
  - `name` (str): Name of the text.
  - `data` (str): Text to write to log file
  - `agent_name` (str): Name of Agent, agent_name is the unique id to identify the workflows of each individual agent in the multi-agent environment.
  - `process_id` (str): The defined process id of agent loop, such as PLAN, ACT, REFLECT, REACT. The process id will be used to group workflow shapes of the agent loop, such as grouping all the workflows in the "PLAN" stage. 


- **Returns**:
  The messages will write one line of log in json dict format with below keys: 
  - `name` (str): Name of the tool/function calling.
  - `data` (str): Json Str of tool schema with arguments type definition.
  - `data_type` (str): text.
  - `timestamp` (str): timestamp of running the function.
  - `workflow_id` (str): an unique id of UUID4() indicating one step in the agent loop workflow.
  - `agent_name` (str): agent_name is used to identify by workflows belonging to which agent.
  - `process_id` (str): process_id is used to group by several workflow_id, such as in PLAN stage, Act stage, etc.


![agentboard text function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_text.jpg?raw=true)


[See full details of ab.summary.text](docs/summary_text.md)


**Logs format for ab.summary.text**

```
{"data": "Please do image search with user input", "name": "Plan Start Prompt", "data_type": "text", "timestamp": 1732092914079, "workflow_id": "5ba41c57-2e44-4442-99c0-8c5e6c0fd0a0", "process_id": "plan", "agent_name": "agent 1"}
```


### ab.summary.dict

- **Description**: Write python dict Data to logs, such as Json of Parameters of tools returned by LLM API, request input, request output, class attributes __dict__.


```
    
ab.summary.dict(
    name: str, 
    data: str, 
    agent_name = None,
    process_id = None, 
    **kwargs
)


```

- **Arguments**:
  - `name` (str): Name of the Dict Data Json converted python dict.
  - `data` (str): Dict Data to write to log file
  - `agent_name` (str): Name of Agent, agent_name is the unique id to identify the workflows of each individual agent in the multi-agent environment.
  - `process_id` (str): The defined process id of agent loop, such as PLAN, ACT, REFLECT, REACT. The process id will be used to group workflow shapes of the agent loop, such as grouping all the workflows in the "PLAN" stage. 


- **Returns**:
  The messages will write one line of log in json dict format with below keys: 

  - `name` (str): Name of the dict.
  - `data` (str): Json Str of tool schema with arguments type definition.
  - `data_type` (str): dict.
  - `timestamp` (str): timestamp of running the function.
  - `workflow_id` (str): an unique id of UUID4() indicating one step in the agent loop workflow.
  - `agent_name` (str): agent_name is used to identify by workflows belonging to which agent.
  - `process_id` (str): process_id is used to group by several workflow_id, such as in PLAN stage, Act stage, etc.


![agentboard dict function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_dict.jpg?raw=true)


[See full details of ab.summary.dict](docs/summary_dict.md)

**Logs format for ab.summary.dict**

```
{"data": "{\"arg1\": 1, \"arg2\": 2}", "name": "Plan Input Args Dict_0", "data_type": "dict", "timestamp": 1732092914079, "workflow_id": "67ae1656-4b9c-4aa5-8054-ec840b190220", "process_id": "plan", "agent_name": "agent 1"}

```



### ab.summary.image


- **Description**: Write Image Tensor Data (pytorch torch.Tensor and tensorflow tf.Tensor) to logs. 


```
    
ab.summary.image(
    name: str, 
    data: Tensor, 
    agent_name = None,
    process_id = None, 
    file_ext = ".png",
    **kwargs
)


```

- **Arguments**:
  - `name` (str): Name of the Image Tensor Data
  - `data` (Tensor): Support Both torch.Tensor and tf.Tensor. torch.Tensor, data takes input tensor shape [N, C, H, W], N: Batch Size, C: Channels, H: Height, W: Width; tf.Tensor, input shape [N, H, W, C], N: Batch Size, H: Height, W: Width, C: Channels.
  - `agent_name` (str): Name of Agent, agent_name is the unique id to identify the workflows of each individual agent in the multi-agent environment.
  - `process_id` (str): The defined process id of agent loop, such as PLAN, ACT, REFLECT, REACT. The process id will be used to group workflow shapes of the agent loop, such as grouping all the workflows in the "PLAN" stage. 
  - `file_ext` (str): The file extension of logged images, possible values of ".png", ".jpg", ".jpeg", etc. The image are saved with pillow Image data format.


- **Returns**:
  The messages will write one line of log in json dict format with below keys: 

  - `name` (str): Name of the image.
  - `data` (str): The output file path of images of tensors in the static file folder for agentboard display.
  - `data_type` (str): image.
  - `timestamp` (str): timestamp of running the function.
  - `workflow_id` (str): an unique id of UUID4() indicating one step in the agent loop workflow.
  - `agent_name` (str): agent_name is used to identify by workflows belonging to which agent.
  - `process_id` (str): process_id is used to group by several workflow_id, such as in PLAN stage, Act stage, etc.


agentboard image visualization

![agentboard image function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_image.jpg?raw=true)

[See full details of ab.summary.image](docs/summary_image.md)

**Logs format for ab.summary.image**

```
{"data": "plan_input_image_0.png", "name": "plan_input_image_0", "data_type": "image", "timestamp": 1732092948859, "workflow_id": "b4b1f770-64a8-48b8-9906-9e6b62fc4198", "process_id": "plan", "agent_name": "agent 1"}
{"data": "plan_input_image_1.png", "name": "plan_input_image_1", "data_type": "image", "timestamp": 1732092948904, "workflow_id": "5db62e21-e0b3-4598-81dc-966597873695", "process_id": "plan", "agent_name": "agent 1"}
{"data": "plan_input_image_2.png", "name": "plan_input_image_2", "data_type": "image", "timestamp": 1732092948949, "workflow_id": "6331c2b0-d802-4446-9f6d-4162871b6efb", "process_id": "plan", "agent_name": "agent 1"}
```



### ab.summary.audio


- **Description**: Write Audio Tensor Data (pytorch torch.Tensor) to logs and export audio files to static folder. 


```
    
ab.summary.audio(
    name: str, 
    data: torch.Tensor, 
    agent_name = None,
    process_id = None,
    file_ext = ".wav",
    sample_rate = 16000,
    **kwargs
)


```

- **Arguments**:
  - `name` (str): Name of the Audio Tensor Data.
  - `data` (Tensor): Support torch.Tensor data type. The input tensor shape [B, C, N], B for batch size, C for channel, N for samples. 
  - `agent_name` (str): Name of Agent, agent_name is the unique id to identify the workflows of each individual agent in the multi-agent environment.
  - `process_id` (str): The defined process id of agent loop, such as PLAN, ACT, REFLECT, REACT. The process id will be used to group workflow shapes of the agent loop, such as grouping all the workflows in the "PLAN" stage. 
  - `file_ext` (str): The file extension of logged images, possible values of ".wav", ".mp3", and others. The audio file are saved using torchaudio.save method.
  - `sample_rate` (int): The sample rate of the audio clip, default to 16000.

- **Returns**:
  The messages will write one line of log in json dict format with below keys: 

  - `name` (str): Name of the Audio Clip.
  - `data` (str): The output file path of audio clips of tensors in the static file folder for agentboard display.
  - `data_type` (str): audio.
  - `timestamp` (str): timestamp of running the function.
  - `workflow_id` (str): an unique id of UUID4() indicating one step in the agent loop workflow.
  - `agent_name` (str): agent_name is used to identify by workflows belonging to which agent.
  - `process_id` (str): process_id is used to group by several workflow_id, such as in PLAN stage, Act stage, etc.


![agentboard audio function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_audio.jpg?raw=true)


[See full details of ab.summary.audio](docs/summary_audio.md)

**Logs format for ab.summary.audio**

```
{"data": "plan_input_audio_0.wav", "name": "plan_input_audio_0", "data_type": "audio", "timestamp": 1732092950091, "workflow_id": "e706e78f-eb92-49b7-8591-e543b01e1d23", "process_id": "plan", "agent_name": "agent 1"}
```

### ab.summary.video


- **Description**: Write Video Tensor Data (pytorch torch.Tensor) to logs and export video files to static folder. 


```
    
ab.summary.video(
    name: str, 
    data: torch.Tensor, 
    agent_name = None,
    process_id = None,
    file_ext = ".mp4",
    frame_rate = 24,
    video_codecs = "mpeg4",
    **kwargs
)


```

- **Arguments**:
  - `name` (str): Name of the Audio Tensor Data.
  - `data` (Tensor): Support torch.Tensor data type. The input tensor shape should match [T, H, W, C], T: Number of frames, H: Height, W: Width, C: Number of channels (usually 3 for RGB)
  - `agent_name` (str): Name of Agent, agent_name is the unique id to identify the workflows of each individual agent in the multi-agent environment.
  - `process_id` (str): The defined process id of agent loop, such as PLAN, ACT, REFLECT, REACT. The process id will be used to group workflow shapes of the agent loop, such as grouping all the workflows in the "PLAN" stage. 
  - `file_ext` (str): The file extension of logged images, possible values of ".mp4", and others. The video file are saved using torchvision.io.write_video method.
  - `frame_rate` (int): The frame rate of the video clip, default to 24.
  - `video_codecs` (str): The video codecs of the video clip, default to "mpeg4". We are using torchvision.io.write_video method to export the video clips


- **Returns**:
  The messages will write one line of log in json dict format with below keys: 

  - `name` (str): Name of the Audio Clip.
  - `data` (str): The output file path of video clips of tensors in the static file folder for agentboard display.
  - `data_type` (str): video.
  - `timestamp` (str): timestamp of running the function.
  - `workflow_id` (str): an unique id of UUID4() indicating one step in the agent loop workflow.
  - `agent_name` (str): agent_name is used to identify by workflows belonging to which agent.
  - `process_id` (str): process_id is used to group by several workflow_id, such as in PLAN stage, Act stage, etc.



![agentboard video function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_loop_workflow_hint.jpg?raw=true)

[See full details of ab.summary.video](docs/summary_video.md)

**Logs format for ab.summary.video**

```
{"data": "demo_video.mp4", "name": "act_output_video", "data_type": "video", "timestamp": 1732092955095, "workflow_id": "553ccba7-a36a-49da-93da-df92a0765051", "process_id": "act", "agent_name": "agent 2"}

```




### `ab.summary.tool.computer_use`



## AgentBoard Supported Web Interface

### 1. X(twitter) Style Social Media Web Admin
### Agent Normal Users



### Agent Website Admin



### Agent Website Automatic Comment Bot







## Agents Related Pipeline Workflow and Document
### AI Services Reviews and Ratings <br>
##### AI Agent
[Microsoft AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-microsoft-ai-agent) <br>
[Claude AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-claude-ai-agent) <br>
[OpenAI AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-openai-ai-agent) <br>
[AgentGPT AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-agentgpt) <br>
[Saleforce AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-salesforce-ai-agent) <br>
[Google AI Agent Reviews](http://www.deepnlp.org/store/pub/pub-google-ai-agent) <br>
##### Chatbot
[OpenAI o1 Reviews](http://www.deepnlp.org/store/pub/pub-openai-o1) <br>
[ChatGPT User Reviews](http://www.deepnlp.org/store/pub/pub-chatgpt-openai) <br>
[Gemini User Reviews](http://www.deepnlp.org/store/pub/pub-gemini-google) <br>
[Perplexity User Reviews](http://www.deepnlp.org/store/pub/pub-perplexity) <br>
[Claude User Reviews](http://www.deepnlp.org/store/pub/pub-claude-anthropic) <br>
[Qwen AI Reviews](http://www.deepnlp.org/store/pub/pub-qwen-alibaba) <br>
[Doubao Reviews](http://www.deepnlp.org/store/pub/pub-doubao-douyin) <br>
[ChatGPT Strawberry](http://www.deepnlp.org/store/pub/pub-chatgpt-strawberry) <br>
[Zhipu AI Reviews](http://www.deepnlp.org/store/pub/pub-zhipu-ai) <br>
##### AI Image Generation
[Midjourney User Reviews](http://www.deepnlp.org/store/pub/pub-midjourney) <br>
[Stable Diffusion User Reviews](http://www.deepnlp.org/store/pub/pub-stable-diffusion) <br>
[Runway User Reviews](http://www.deepnlp.org/store/pub/pub-runway) <br>
[GPT-5 Forecast](http://www.deepnlp.org/store/pub/pub-gpt-5) <br>
[Flux AI Reviews](http://www.deepnlp.org/store/pub/pub-flux-1-black-forest-lab) <br>
[Canva User Reviews](http://www.deepnlp.org/store/pub/pub-canva) <br>
##### AI Video Generation
[Luma AI](http://www.deepnlp.org/store/pub/pub-luma-ai) <br>
[Pika AI Reviews](http://www.deepnlp.org/store/pub/pub-pika) <br>
[Runway AI Reviews](http://www.deepnlp.org/store/pub/pub-runway) <br>
[Kling AI Reviews](http://www.deepnlp.org/store/pub/pub-kling-kwai) <br>
[Dreamina AI Reviews](http://www.deepnlp.org/store/pub/pub-dreamina-douyin) <br>
##### AI Education
[Coursera Reviews](http://www.deepnlp.org/store/pub/pub-coursera) <br>
[Udacity Reviews](http://www.deepnlp.org/store/pub/pub-udacity) <br>
[Grammarly Reviews](http://www.deepnlp.org/store/pub/pub-grammarly) <br>
##### Robotics
[Tesla Cybercab Robotaxi](http://www.deepnlp.org/store/pub/pub-tesla-cybercab) <br>
[Tesla Optimus](http://www.deepnlp.org/store/pub/pub-tesla-optimus) <br>
[Figure AI](http://www.deepnlp.org/store/pub/pub-figure-ai) <br>
[Unitree Robotics Reviews](http://www.deepnlp.org/store/pub/pub-unitree-robotics) <br>
[Waymo User Reviews](http://www.deepnlp.org/store/pub/pub-waymo-google) <br>
[ANYbotics Reviews](http://www.deepnlp.org/store/pub/pub-anybotics) <br>
[Boston Dynamics](http://www.deepnlp.org/store/pub/pub-boston-dynamic) <br>
##### AI Tools
[DeepNLP AI Tools](http://www.deepnlp.org/store/pub/pub-deepnlp-ai) <br>
##### AI Widgets
[Apple Glasses](http://www.deepnlp.org/store/pub/pub-apple-glasses) <br>
[Meta Glasses](http://www.deepnlp.org/store/pub/pub-meta-glasses) <br>
[Apple AR VR Headset](http://www.deepnlp.org/store/pub/pub-apple-ar-vr-headset) <br>
[Google Glass](http://www.deepnlp.org/store/pub/pub-google-glass) <br>
[Meta VR Headset](http://www.deepnlp.org/store/pub/pub-meta-vr-headset) <br>
[Google AR VR Headsets](http://www.deepnlp.org/store/pub/pub-google-ar-vr-headset) <br>
##### Social
[Character AI](http://www.deepnlp.org/store/pub/pub-character-ai) <br>
##### Self-Driving
[BYD Seal](http://www.deepnlp.org/store/pub/pub-byd-seal) <br>
[Tesla Model 3](http://www.deepnlp.org/store/pub/pub-tesla-model-3) <br>
[BMW i4](http://www.deepnlp.org/store/pub/pub-bmw-i4) <br>
[Baidu Apollo Reviews](http://www.deepnlp.org/store/pub/pub-baidu-apollo) <br>
[Hyundai IONIQ 6](http://www.deepnlp.org/store/pub/pub-hyundai-ioniq-6) <br>


### Related Blogs <br>
[DeepNLP AI Agents Designing Guidelines](http://www.deepnlp.org/blog?category=agent) <br>
[Dialogue Visualization Agent Multimodal Visualization Tools for AI Systems A Review](http://www.deepnlp.org/blog/dialogue-agent-multimodal-visualization-tools-for-ai-systems) <br>
[AI Agent Visualization Review Asynchronous Multi-Agent Simulation](http://www.deepnlp.org/blog/ai-agent-visualization-review-asynchronous-multi-agent-simulation) <br>
[Tencent Wechat Public Account AI Agents Tutorial Finance Agent Example](http://www.deepnlp.org/blog/tencent-wechat-public-account-ai-agents-tutorial-a-finance-agent-example) <br>
[AgentBoard AI Agent Visualization Tutorial](http://www.deepnlp.org/blog/agentBoard-ai-agent-visualization-toolkit-agent-loop-workflow) <br>
[AI Agent User Review](http://www.deepnlp.org/store/ai-agent) <br>
[Introduction to multimodal generative models](http://www.deepnlp.org/blog/introduction-to-multimodal-generative-models) <br>
[Generative AI Search Engine Optimization](http://www.deepnlp.org/blog/generative-ai-search-engine-optimization-how-to-improve-your-content) <br>
[AI Image Generator User Reviews](http://www.deepnlp.org/store/image-generator) <br>
[AI Video Generator User Reviews](http://www.deepnlp.org/store/video-generator) <br>
[AI Chatbot & Assistant Reviews](http://www.deepnlp.org/store/chatbot-assistant) <br>
[Best AI Tools User Reviews](http://www.deepnlp.org/store/pub/) <br>
[AI Boyfriend User Reviews](http://www.deepnlp.org/store/chatbot-assistant/ai-boyfriend) <br>
[AI Girlfriend User Reviews](http://www.deepnlp.org/store/chatbot-assistant/ai-girlfriend) <br>

