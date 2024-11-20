# AgentBoard: AI Agent Visualization Toolkit Board Document

DeepNLP AgentBoard is a visualization toolskit (similar to Tensorboard to visualize tensors) to visualize and monitor the agent loops and key entities of AI Agents development, such as messages, tools/functions, workflow and raw data types including text, dict or json, image, audio, video, etc. You can easily add logs with agentboard together with various AI agent frameworks, such as AutoGen, Langgraph, AutoAgent. 


You can install and import the 'agentboard' python package and use functions under a with block. See quickstart for install and run agentboard [See full details of Quickstart](docs/quickstart.md).

## AgentBoard Supported AI Agent Loop Elements and Data Types

|  Functions  | DataType |  Description  |
|  -------- | --------  | --------  |
|  [**ab.summary.messages**](#absummarymessages) | message |   List of messages, json format [{"role": "user", "content": "content_1"}, {"role": "assistant", "content": "content_2"}] |
|  [**ab.summary.tool**](#absummarytool) |  function |   User defined functions, The schema of the functions which are passed to LLM API calling, Support OpenAI and Anthropic stype  |
|  [**ab.summary.text**](#absummarytext)  |  str |   Text data, such as prompt, assistant responded text  |
|  [**ab.summary.dict**](#absummarydict)  |  dict  |   Dict data, such as input request, output response, class __dict__ |
|  [**ab.summary.image**](#absummaryimage)  | tensor |   Support both torch.Tensor and tf.Tensor,  torch.Tensor takes input shape [N, C, H, W], N: Batch Size, C: Channels, H: Height, W: Width; tf.Tensor, input shape [N, H, W, C], N: Batch Size, H: Height, W: Width, C: Channels.  |
|  [**ab.summary.audio**](#absummaryaudio)   | tensor |  Support torch.Tensor data type. The input tensor shape [B, C, N], B for batch size, C for channel, N for samples. |
|  [**ab.summary.video**](#absummaryvideo)  | tensor |  Support torch.Tensor data type. The input tensor shape should match [T, H, W, C], T: Number of frames, H: Height, W: Width, C: Number of channels (usually 3 for RGB) |


### `ab.summary.messages`
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
  - `data_type` (str): message.
  - `timestamp` (str): timestamp of running the function.
  - `workflow_id` (str): an unique id of UUID4() indicating one step in the agent loop workflow.
  - `agent_name` (str): agent_name is used to identify by workflows belonging to which agent.
  - `process_id` (str): process_id is used to group by several workflow_id, such as in PLAN stage, Act stage, etc.


agentboard messages visualization as chat

![agentboard summary messages function]()


[See full details of ab.summary.messages](docs/summary_messages.md)




### `ab.summary.tool`
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


agentboard tool visualization

![agentboard tool function]()


[See full details of ab.summary.tool](docs/summary_tool.md)



### `ab.summary.text`
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


agentboard text visualization

![agentboard text messages function]()


[See full details of ab.summary.text](docs/summary_text.md)


### `ab.summary.dict`

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


agentboard dict visualization

![agentboard dict messages function]()


[See full details of ab.summary.dict](docs/summary_dict.md)



### `ab.summary.image`


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

![agentboard image messages function]()


[See full details of ab.summary.image](docs/summary_image.md)


### `ab.summary.audio`


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


agentboard audio visualization

![agentboard audio messages function]()


[See full details of ab.summary.audio](docs/summary_audio.md)



### `ab.summary.video`


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


agentboard video visualization

![agentboard video messages function]()

[See full details of ab.summary.video](docs/summary_video.md)



