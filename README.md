# AgentBoard: AI Agent Visualization Toolkit 

DeepNLP AgentBoard provides the toolkit of visualization of AI Agents Worflow, which helps visualize and monitor the AI Agents running loop and key entities such as chat messages, tools/functions, RAG(Retrieval Augmented Generation), workflow and raw data types including text, dict or json, image, audio, video, etc. 

![agentboard agent loop workflow](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_loop_workflow_hint.jpg?raw=true)


## Key Features
- APIs to log various data types for LLM calling and AI Agent development: text, dict or json, image, audio, video, etc. 
- AI Agent Key Entities Visualization: Visualize key entities of chat history messages, memories, tools/functions schema and input/output dict, etc.
- Workflow of AI Agent Loop Running, Visualize the plan/act/react/reflect stage in the complete workflow chart. Can be modified and displayed for papers and technical reports.
- Multi-Agents Support
- Autonomous Agent Simulation Enviroment Support. Multiple build-in AI agent environment AI community, X(twitter) style mininal website for AI agents to interact (comment/like/follow/etc). 
- Chat Visualizer: A Chatbot Visualizer with multiple mainstream chat(WhatsApp, WeChat, ChatGPT, etc.) UI

You can install and import the 'agentboard' python package and use functions under a with block. See quickstart for install and run agentboard 

<br>

[DeepNLP AgentBoard Quickstart Docs](docs/quickstart.md) <br>

[DeepNLP AgentBoard Python Full API Docs](docs/agentboard_docs.md)


## AgentBoard Supported AI Agent Elements and Data Types

|  Functions  | DataType |  Description  |
|  -------- | --------  | --------  |
|  [**ab.summary.messages**](#absummarymessages) | message |   List of messages, json format [{"role": "user", "content": "content_1"}, {"role": "assistant", "content": "content_2"}] |
|  [**ab.summary.tool**](#absummarytool) |  function |   User defined functions, The schema of the functions which are passed to LLM API calling, Support OpenAI and Anthropic stype  |
|  [**ab.summary.agent_loop**](#absummaryagentloop) |  str |   User defined Agent Loop with various stages PLAN/ACT/REFLECT/etc  |
|  [**ab.summary.text**](#absummarytext)  |  str |   Text data, such as prompt, assistant responded text  |
|  [**ab.summary.dict**](#absummarydict)  |  dict  |   Dict data, such as input request, output response, class __dict__ |
|  [**ab.summary.image**](#absummaryimage)  | tensor |   Support both torch.Tensor and tf.Tensor,  torch.Tensor takes input shape [N, C, H, W], N: Batch Size, C: Channels, H: Height, W: Width; tf.Tensor, input shape [N, H, W, C], N: Batch Size, H: Height, W: Width, C: Channels.  |
|  [**ab.summary.audio**](#absummaryaudio)   | tensor |  Support torch.Tensor data type. The input tensor shape [B, C, N], B for batch size, C for channel, N for samples. |
|  [**ab.summary.video**](#absummaryvideo)  | tensor |  Support torch.Tensor data type. The input tensor shape should match [T, H, W, C], T: Number of frames, H: Height, W: Width, C: Number of channels (usually 3 for RGB) |

## Installation

You can install agentboard through pip and start the Flask based web app using the command line agentboard. You can change 'logdir', 'logfile', 'static' and 'port' parameters to point to your folders. 
After installation, you can visit (http://127.0.0.1:5000) to see the web console of agentboard. There is build in log file to visualize multiple data types.

```
# installation
pip install agentboard

# cmd line start service
agentboard

# Change log dir and port agentboard --logdir=./log --logfile=xxx.log --static=./static --port=5000
```


## AgentBoard Log Messages Chat Visualizer

Let's start with an example of calling OpenAI LLM api with user input prompt.

```

    import agentboard as ab

    messages= []
    logdir="./log"
    with ab.summary.FileWriter(logdir=logdir) as writer:
        prompt = "Can you give me an example of python code usage of async await functions"
        messages.append({"role": "user", "content": prompt})

        ## Calling OpenAI Chat Completion API            
        #        completion = client.chat.completions.create(
        #            model="gpt-3.5-turbo",
        #            messages=[
        #                {"role": "system", "content": "You are a helpful assistant."},
        #                {"role": "user", "content": prompt}
        #            ]
        #        )

        response_message = {"role":"assistant","content":"Sure! Here's an example of Python code that uses async/await functions:\n\n```python\nimport asyncio\n\nasync def print_numbers():\n    for i in range(1, 6):\n        print(i)\n        await asyncio.sleep(1)\n\nasync def main():\n    task1 = asyncio.create_task(print_numbers())\n    task2 = asyncio.create_task(print_numbers())\n    await task1\n    await task2\n\nasyncio.run(main())\n```\n\nIn this example, we define an async function `print_numbers` that prints the numbers 1 to 5 with a one-second delay between each number using `await asyncio.sleep(1)`. We also define an async function `main` that creates two tasks using `asyncio.create_task` to run the `print_numbers` function concurrently. We then use `await` to wait for both tasks to complete.\n\nWhen we run the `main` function using `asyncio.run(main())`, the numbers will be printed concurrently by the two tasks."}

        messages.append(response_message)

        ab.summary.messages(name="OpenAI Chat History", data=messages, agent_name="assistant")

```

Then you can go visit the agentboard (http://127.0.0.1:5000/log/messages) to see the chat visualizer of the chat completion history. 
* Note the log writing --logdir and agentboard loading --logdir should match.


![agentboard summary messages function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_chat_visualizer.jpg?raw=true)


## AgentBoard Agent Loop Visualization

Let's start with an example of a basic asynchronously AI Agent Loop, consists of 3 stages: PLAN, ACT, REFLECT. 
And plot the stage and result on a workflow chart on agentboard. We can use the basic Asynchronous Agent Loop in the AutoAgent Package, you can also use agentboard with many other agent framework, such as AutoGen and LangChain.


To use agentboard with basic Async Agent Class, we need to first rewrtie the agent with logger in the desired place.


```
    cd /examples/async_agents/

    # write logs and static file, default to ./log and ./static
    python run_agentboard_autoagent.py

    # run agentboard and visualize agent loop
    agentboard --logdir=./log --static=./static --port=5000

```

visit the agentboard (http://127.0.0.1:5000/log/agent_loop) 

![agentboard agent loop workflow](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_loop_workflow_hint.jpg?raw=true)


## AgentBoard to log Tools Function Calls

Let's start with an example of calling OpenAI Tool Usage API with user defined functions get_weather().

```


    import agentboard as ab
    from agentboard.utils import function_to_schema

    def calling_bing_tools(keyword:str, limit:int) -> str:
        url="https://www.bing.com/search?q=%s&limit=%d" % (keyword, limit)
        return url

    with ab.summary.FileWriter(logdir="./log", static="./static") as writer:

        tools = [calling_bing_tools]
        tools_map = {tool.__name__:tool for tool in tools}
        tools_schema = [function_to_schema(tool) for tool in tools]

        ## Calling ChatGPT Tool Usage code omitted
        # Omitted
        # arguments = json.loads(tool_call['function']['arguments'])

        arguments = {"keyword": "agentboard document", "limit": 10}


        ab.summary.tool(name="Act RAG Tool Bing", data=[calling_bing_tools], agent_name="agent 2", process_id="ACT")
        ab.summary.dict(name="Act RAG Argument Input", data=[arguments], agent_name="agent 2", process_id="ACT")

```


![agentboard tool function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_tool.jpg?raw=true)


## AgentBoard to Display Image Tensor

Let's log a random pytorch tensor with shape [8, 3, 400, 600] and display the batch image tensors on the agentboard.

```
import torch
import agentboard as ab

with ab.summary.FileWriter(logdir="./log", static="./static") as writer:

    input_image = torch.mul(torch.rand(8, 3, 400, 600), 255).to(torch.int64)
    ab.summary.image(name="Plan Input Image", data=input_image, agent_name="agent 1", process_id="plan")
```

![agentboard image function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_image.jpg?raw=true)


## AgentBoard Display Audio Tensor

Let's log a random pytorch tensor of audio with 2 channels, 16000 sample_rate lasting for 2 seconds.


```
import torch
import agentboard as ab
import math

with ab.summary.FileWriter(logdir="./log", static="./static") as writer:
    sample_rate = 16000  # 16 kHz
    duration_seconds = 2  # 2 seconds
    frequency = 440.0  # 440 Hz (A4 note)
    t = torch.linspace(0, duration_seconds, int(sample_rate * duration_seconds), dtype=torch.float32)
    waveform = (0.5 * torch.sin(2 * math.pi * frequency * t)).unsqueeze(0)  # Add channel dimension
    waveform = torch.unsqueeze(waveform, dim=0)
    ab.summary.audio(name="ASR Input Audio", data=waveform, agent_name="agent 1", process_id="asr")
```

![agentboard audio function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_audio.jpg?raw=true)


## AgentBoard Display Video Tensor

Let's log a random pytorch tensor of video clip with 30 frames, 64x64 resolution, 3 color channels and use agentboard to visualize it.


```
import torch
import agentboard as ab

with ab.summary.FileWriter(logdir="./log", static="./static") as writer:
    T, H, W, C = 30, 64, 64, 3  # 30 frames, 64x64 resolution, 3 color channels
    video_tensor = torch.randint(0, 256, (T, H, W, C), dtype=torch.uint8)
    frame_rate = 24  # Frames per second
    ab.summary.video(name="Text2Video Output", data=video_tensor, agent_name="agent 1", 
        process_id="act", file_ext = ".mp4", frame_rate = 24, video_codecs = "mpeg4")

```

![agentboard audio function](https://github.com/AI-Hub-Admin/agentboard/blob/main/docs/demo_agentboard_video.jpg?raw=true)




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
[AgentBoard AI Agent Visualization Toolkit](http://www.deepnlp.org/blog/agentboard-ai-agent-visualization-toolkit-agent-loop-workflow) <br>
[DeepNLP AI Agents Designing Guidelines](http://www.deepnlp.org/blog?category=agent) <br>
[Introduction to multimodal generative models](http://www.deepnlp.org/blog/introduction-to-multimodal-generative-models) <br>
[Generative AI Search Engine Optimization](http://www.deepnlp.org/blog/generative-ai-search-engine-optimization-how-to-improve-your-content) <br>
[AI Image Generator User Reviews](http://www.deepnlp.org/store/image-generator) <br>
[AI Video Generator User Reviews](http://www.deepnlp.org/store/video-generator) <br>
[AI Chatbot & Assistant Reviews](http://www.deepnlp.org/store/chatbot-assistant) <br>
[Best AI Tools User Reviews](http://www.deepnlp.org/store/pub/) <br>
[AI Boyfriend User Reviews](http://www.deepnlp.org/store/chatbot-assistant/ai-boyfriend) <br>
[AI Girlfriend User Reviews](http://www.deepnlp.org/store/chatbot-assistant/ai-girlfriend) <br>

