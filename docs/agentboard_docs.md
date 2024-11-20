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

![agentboard summary messages function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_chat_visualizer.jpg?raw=true)

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

![agentboard tool function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_tool.jpg?raw=true)


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

![agentboard text function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_text.jpg?raw=true)


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

![agentboard dict function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_dict.jpg?raw=true)


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

![agentboard image messages function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_image.jpg?raw=true)


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

![agentboard audio messages function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_audio.jpg?raw=true)


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

![agentboard video messages function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_video.jpg?raw=true)

[See full details of ab.summary.video](docs/summary_video.md)




## Agents Related Pipeline Workflow and Document
### AI Services Reviews and Ratings <br>
##### AI Agent
[Microsoft AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-microsoft-ai-agent) <br>
[Claude AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-claude-ai-agent) <br>
[OpenAI AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-openai-ai-agent) <br>
[AgentGPT AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-agentgpt) <br>
[Saleforce AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-salesforce-ai-agent) <br>
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
[Introduction to multimodal generative models](http://www.deepnlp.org/blog/introduction-to-multimodal-generative-models) <br>
[Generative AI Search Engine Optimization](http://www.deepnlp.org/blog/generative-ai-search-engine-optimization-how-to-improve-your-content) <br>
[AI Image Generator User Reviews](http://www.deepnlp.org/store/image-generator) <br>
[AI Video Generator User Reviews](http://www.deepnlp.org/store/video-generator) <br>
[AI Chatbot & Assistant Reviews](http://www.deepnlp.org/store/chatbot-assistant) <br>
[Best AI Tools User Reviews](http://www.deepnlp.org/store/pub/) <br>
[AI Boyfriend User Reviews](http://www.deepnlp.org/store/chatbot-assistant/ai-boyfriend) <br>
[AI Girlfriend User Reviews](http://www.deepnlp.org/store/chatbot-assistant/ai-girlfriend) <br>

