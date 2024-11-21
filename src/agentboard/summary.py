# -*- coding: utf-8 -*-
# @Time    : 2024/06/27

import json
import datetime
import os
import numpy as np
import queue
from queue import Queue
import contextvars
import math
import traceback
from inspect import isfunction

from .core_constants import *
from .utils import *

context_file_writer = contextvars.ContextVar('context_file_writer')

class FileWriter(object):

    def __init__(self, logdir, static="./static", flush_secs=120, **kwargs):
        self._logdir = str(logdir)
        if not os.path.exists(self._logdir):
            os.mkdir(self._logdir)
        self._static_dir = str(static)
        if not os.path.exists(self._static_dir):
            os.mkdir(self._static_dir)
        self._initialize()

        ## flush timing
        self._flush_secs = flush_secs
        # The first event will be flushed immediately.
        self._next_event_flush_time = 0
        self.debug = kwargs["debug"] if "debug" in kwargs else False

        ## set file write to this
        context_file_writer.set(self)

    def get_log_file_name(self):
        """
            Internal Log File name Format 
        """
        now = datetime.datetime.now()
        filename = "{year}{month}{day}{hour}{minute}{second}.log".format(year=now.year, month=now.month, day = now.day, hour = now.hour, minute=now.minute, second= now.second)
        filepath = os.path.join(self.get_logdir(), filename)
        return filepath

    def _initialize(self):
        self._max_queue_size = 1000
        self._log_queue = Queue(maxsize = self._max_queue_size)
        print ("DEBUG: Initialize File Write Log Queue size 1000....")

        # initialize a file write
        log_file_name = self.get_log_file_name()
        self.log_file = open(log_file_name, 'w')

        while self._log_queue.qsize() == self._max_queue_size:
            # Todo write to disk
            print ("_log_queue is full...")

    def add_log(self, log_data):
        """
            log_data: dict
        """
        self._log_queue.put(log_data)

    def write_log(self):
        """
        """
        if self.log_file is None:
            print("DEBUG: Log File is null...")
            return
        cur_queue_size = self._log_queue.qsize()
        while not self._log_queue.empty():
            data_dict = self._log_queue.get()
            if data_dict is not None:
                self.log_file.writelines([json.dumps(data_dict) + "\n"])
        if self.debug:
            print ("DEBUG: Write Logs in Queue Cnt %d to file %s" % (cur_queue_size, self.log_file.name))

    def save_image(self, filename, img):
        """ 
            Save image to local file
            filename: str
            data: PIL Image Class
        """
        try:
            import PIL
            from PIL import Image

            img_dir = os.path.join(self._static_dir, "img")
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            if isinstance(img, Image.Image):
                img_path = os.path.join(img_dir, filename)        
                output = img.save(img_path)
            else:
                print ("DEBUG: Writer Save Image input not supported")
        except Exception as e:
            print (e)
            s = traceback.format_exc()
            print (s)

    def save_audio(self, filename, waveform, sample_rate=16000):
        """
            waveform: torch.Tensor
        """
        try:
            import torchaudio
            audio_dir = os.path.join(self._static_dir, "audio")
            if not os.path.exists(audio_dir):
                os.mkdir(audio_dir)
            audio_path = os.path.join(audio_dir, filename)        
            torchaudio.save(audio_path, waveform, sample_rate)
        except Exception as e:
            print (e)
            s = traceback.format_exc()
            print (s)

    def save_video(self, filename, video, frame_rate=24, video_codec="mpeg4"):

        """
            filename: str,
            video: torch tensor with shape [T, H, W, C], 
                T: Number of frames
                H: Height
                W: Width
                C: Number of channels (usually 3 for RGB)
            frame_rate: int, default : 24
            video_codec: str, default : mpeg4
        """
        try:
            import torchvision.io as io
            video_dir = os.path.join(self._static_dir, "video")
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            video_path = os.path.join(video_dir, filename)        
            io.write_video(video_path, video, fps=frame_rate, video_codec=video_codec)
        except Exception as e:
            print (e)
            s = traceback.format_exc()
            print (s)

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """Make usable with "with" statement."""
        if self.log_file is not None:
            self.log_file.close()

def text(name, data, **kwargs):
    """ AgentBoard Add Process
    """
    ## write logs
    writer = context_file_writer.get("context_file_writer")
    try:
        if not isinstance(data, str):
            print ("ERROR: text input data should be str format")
            return
        log_data = {}
        log_data[KEY_DATA] = data
        log_data[KEY_NAME] = name
        log_data[KEY_DATA_TYPE] = DATA_TYPE_TEXT
        log_data[KEY_TIMESTAMP] = get_current_timestamp_milli()
        log_data[KEY_WORKFLOW_ID] = get_workflow_id()
        ## required fields
        log_data[KEY_PROCESS_ID] = kwargs[KEY_PROCESS_ID] if KEY_PROCESS_ID in kwargs else ""
        log_data[KEY_AGENT_NAME] = kwargs[KEY_AGENT_NAME] if KEY_AGENT_NAME in kwargs else ""

        ## default values
        log_data.update(kwargs)
        if writer is not None:
            writer.add_log(log_data)
            writer.write_log()
        else:
            print("DEBUG: FileWriter is missing in contextvars...")
    except Exception as e:
        print (e)
        s = traceback.format_exc()
        print (s)

def dict(name, data, **kwargs):
    """ AgentBoard Add Process
        data: list of dict
    """
    ## write logs
    import json
    writer = context_file_writer.get("context_file_writer")
    if writer is None:
        print ("DEBUG: Summary Writer is None...")
        return
    try:
        if not isinstance(data, list):
            print ("ERROR: Input Data is not in python list of dict format")
            return
        if len(data) == 0:
            print ("ERROR: Input Data dict list is empty has len 0...")
            return
        log_data_list = []
        for i in range(len(data)):
            log_data = {}
            log_data[KEY_DATA] = json.dumps(data[i])
            log_data[KEY_NAME] = name + "_" + str(i)            
            log_data[KEY_DATA_TYPE] = DATA_TYPE_DICT
            log_data[KEY_TIMESTAMP] = get_current_timestamp_milli()
            log_data[KEY_WORKFLOW_ID] = get_workflow_id()
            ## required fields
            log_data[KEY_PROCESS_ID] = kwargs[KEY_PROCESS_ID] if KEY_PROCESS_ID in kwargs else ""
            log_data[KEY_AGENT_NAME] = kwargs[KEY_AGENT_NAME] if KEY_AGENT_NAME in kwargs else ""
            ## default values
            log_data.update(kwargs)
            log_data_list.append(log_data)

        # write to log data     
        [writer.add_log(log_data) for log_data in log_data_list]
        writer.write_log()
    except Exception as e:
        print (e)
        s = traceback.format_exc()
        print (s)

def image(name, data, **kwargs):
    """

        data: 
            tf.Tensor, shape [N, H, W, C], N: Batch Size, H: Height, W: Width, C: Channels
            torch.Tensor, shape [N, C, H, W], N: Batch Size, C: Channels, H: Height, W: Width
    """
    writer = context_file_writer.get("context_file_writer")    
    if writer is None:
        print ("DEBUG: Summary Writer is None...")
        return
    try:
        import torch
        import tensorflow as tf
        from PIL import Image
        # image numpy shape [N, C, H, W]
        image_data_numpy = None
        file_ext = kwargs[KEY_FILE_EXT] if KEY_FILE_EXT in kwargs else DEFAULT_IMAGE_EXT

        if isinstance(data, tf.Tensor):
            assert len(data.shape) == 4, "Input Tensorflow tensor shape should be 4 dimensions, [B, W, H, C], B for batch size, C for channel, W for width, H for height"            
            version = tf.__version__
            if version.startswith('1'):
                with tf.Session() as sess:
                    image_data_numpy = data.eval()
            elif version.startswith('2'):
                image_data_numpy = data.numpy()
            else:
                print("ERROR: Unknown TensorFlow version.")
            # [N, H, W, C] -> [N, C, H, W]
            image_data_numpy = image_data_numpy.transpose(0, 3, 1, 2)
        elif isinstance(data, torch.Tensor):
            assert len(data.shape) == 4, "Input data shape of Pytorch tensor should be in [B, C, W, H], B for batch size, C for channel, W for width, H for height"
            image_data_numpy = data.detach().cpu().numpy()
        else:
            print ("ERROR: image input data type not supported %s" % str(type(data)))
            return 

        assert len(image_data_numpy.shape) == 4, "Converted Numpy tensors for Images should be 4 dimensions [B, C, W, H], B for batch size, C for channel, W for width, H for height"        

        log_data_list = []
        # save to local file
        batch_size = image_data_numpy.shape[0]
        for i in range(batch_size):
            image_array = image_data_numpy[i]
            image_array = np.clip(image_array, 0, 255)
            # [W, H, C]
            image_array = np.transpose(image_array, (1, 2, 0))
            image_array = image_array.astype(np.uint8)
            # Save Image file to local
            image_name = normalize_data_name(name) + "_" + str(i)
            filename = image_name + file_ext
            writer.save_image(filename, Image.fromarray(image_array))
            # write to log data
            log_data = {}
            log_data[KEY_DATA] = filename
            log_data[KEY_NAME] = image_name
            log_data[KEY_DATA_TYPE] = DATA_TYPE_IMAGE
            log_data[KEY_TIMESTAMP] = get_current_timestamp_milli()
            log_data[KEY_WORKFLOW_ID] = get_workflow_id()
            ## required fields
            log_data[KEY_PROCESS_ID] = kwargs[KEY_PROCESS_ID] if KEY_PROCESS_ID in kwargs else ""
            log_data[KEY_AGENT_NAME] = kwargs[KEY_AGENT_NAME] if KEY_AGENT_NAME in kwargs else ""
            log_data_list.append(log_data)
        # write to log data     
        [writer.add_log(log_data) for log_data in log_data_list]
        writer.write_log()
    except Exception as e:
        print (e)
        s = traceback.format_exc()
        print (s)

def video(name, data, **kwargs):
    """
        data: one tensor representing video with shape (T, H, W, C) T: Number of frames,  H: Height, W: Width, C: Number of channels (usually 3 for RGB)

        See Ref: https://pytorch.org/vision/main/generated/torchvision.io.write_video.html
        (T, H, W, C) T: Number of frames,  H: Height, W: Width, C: Number of channels (usually 3 for RGB)
    """
    writer = context_file_writer.get("context_file_writer")
    if writer is None:
        print ("DEBUG: Summary Writer is None...")
        return
    try:
        import torch
        import tensorflow as tf
        import torchvision
        if not isinstance(data, torch.Tensor):
            print ("DEBUG: Summary Video input data is not pytorch tensor type...")
            return

        assert len(data.shape) == 4, "Input Data tensor for video should be 4 dimensions (T, H, W, C)."
        file_ext = kwargs[KEY_FILE_EXT] if KEY_FILE_EXT in kwargs else DEFAULT_VIDEO_EXT
        frame_rate = kwargs[KEY_FRAME_RATE] if KEY_FRAME_RATE in kwargs else DEFAULT_FRAME_RATE
        video_codec = kwargs[KEY_VIDEO_CODECS] if KEY_VIDEO_CODECS in kwargs else DEFAULT_VIDEO_CODECS
        # write to local file
        video_name = normalize_data_name(name)
        filename = video_name + file_ext
        writer.save_video(filename, data, frame_rate=frame_rate, video_codec=video_codec)
        # write to log data
        log_data_list = []
        log_data = {}
        ## image data save file_path
        log_data[KEY_DATA] = filename
        log_data[KEY_NAME] = video_name
        log_data[KEY_DATA_TYPE] = DATA_TYPE_VIDEO
        log_data[KEY_TIMESTAMP] = get_current_timestamp_milli()
        log_data[KEY_WORKFLOW_ID] = get_workflow_id()
        ## required fields
        log_data[KEY_PROCESS_ID] = kwargs[KEY_PROCESS_ID] if KEY_PROCESS_ID in kwargs else ""
        log_data[KEY_AGENT_NAME] = kwargs[KEY_AGENT_NAME] if KEY_AGENT_NAME in kwargs else ""
        log_data_list.append(log_data)
        ## log data
        [writer.add_log(log_data) for log_data in log_data_list]
        writer.write_log()
    except Exception as e:
        print (e)
        s = traceback.format_exc()
        print (s)

def audio(name, data, **kwargs):
    """
        See Ref: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html 
        audio tensor shape [B, C, N], B for batch size, C for channel, N for samples
    """
    writer = context_file_writer.get("context_file_writer")
    if writer is None:
        print ("DEBUG: Summary Writer is None...")
        return
    try:
        import torch
        import torchaudio

        if not isinstance(data, torch.Tensor):
            print ("ERROR: summary.audio Input tensor is not of type torch.Tensor")
            return

        assert len(data.shape) == 3, "Input data shape of Pytorch tensor should be in [B, C, N], B for batch size, C for channel, N for samples"

        file_ext = kwargs[KEY_FILE_EXT] if KEY_FILE_EXT in kwargs else DEFAULT_AUDIO_EXT            
        sample_rate = kwargs[KEY_AUDIO_SAMPLE_RATE] if KEY_AUDIO_SAMPLE_RATE in kwargs else DEFAULT_AUDIO_SAMPLE_RATE

        batch_size = data.shape[0]
        log_data_list = []
        for i in range(batch_size):
            # [C, H, W]
            audio_name = normalize_data_name(name) + "_" + str(i)
            filename = audio_name + file_ext
            # write to local file
            writer.save_audio(filename, data[i], sample_rate)
            # write to log data
            log_data = {}
            ## image data save file_path
            log_data[KEY_DATA] = filename
            log_data[KEY_NAME] = audio_name
            log_data[KEY_DATA_TYPE] = DATA_TYPE_AUDIO
            log_data[KEY_TIMESTAMP] = get_current_timestamp_milli()
            log_data[KEY_WORKFLOW_ID] = get_workflow_id()
            ## required fields
            log_data[KEY_PROCESS_ID] = kwargs[KEY_PROCESS_ID] if KEY_PROCESS_ID in kwargs else ""
            log_data[KEY_AGENT_NAME] = kwargs[KEY_AGENT_NAME] if KEY_AGENT_NAME in kwargs else ""
            log_data_list.append(log_data)
            
        # write to logs
        [writer.add_log(log_data) for log_data in log_data_list]
        writer.write_log()
    except Exception as e:
        print (e)
        s = traceback.format_exc()
        print (s)


### Agent Tool Summary Function
def tool(name, data, **kwargs):
    """
        name: str
        data: list of python function
    """
    writer = context_file_writer.get("context_file_writer")
    if writer is None:
        print ("DEBUG: Summary Writer is None...")
        return
    try:        
        assert isinstance(data, list), "Input Data format for tool should be list of functions"
        if len(data) == 0:
            return 
        assert isfunction(data[0]), "Input Data format for tool should be function%s" % str(data[0])
        log_data_list = []
        func_num = len(data)
        for i in range(func_num):
            func = data[i]
            # write to log data
            log_data = {}
            ## image data save file_path, data: function schema, name: function name
            log_data[KEY_DATA] = json.dumps(function_to_schema(func))
            log_data[KEY_NAME] = func.__name__
            log_data[KEY_DATA_TYPE] = DATA_TYPE_TOOL
            log_data[KEY_TIMESTAMP] = get_current_timestamp_milli()
            log_data[KEY_WORKFLOW_ID] = get_workflow_id()
            ## required fields
            log_data[KEY_PROCESS_ID] = kwargs[KEY_PROCESS_ID] if KEY_PROCESS_ID in kwargs else ""
            log_data[KEY_AGENT_NAME] = kwargs[KEY_AGENT_NAME] if KEY_AGENT_NAME in kwargs else ""
            
            log_data.update(kwargs)
            log_data_list.append(log_data)
        ## add log to writer       
        [writer.add_log(log_data) for log_data in log_data_list]
        writer.write_log()
    except Exception as e:
        print (e)
        s = traceback.format_exc()
        print (s)



### Agent Tool Summary Function
def messages(name, data, **kwargs):
    """
        name: str
        data: list of messages dict, format:  [{"role":"user", "content": "hello"}, {"role":"assistant", "content": "Hola!"}], output 1 line of log
    """
    writer = context_file_writer.get("context_file_writer")
    if writer is None:
        print ("DEBUG: Summary Writer for messages is None...")
        return
    try:        
        assert isinstance(data, list), "Input Data format for messages should be list of dict of role, content "
        if len(data) == 0:
            return 
        # assert isinstance(data[0], dict), "Input Data format for messages should be dict %s" % str(data[0])
        log_data_list = []

        # write to log data
        log_data = {}
        ## image data save file_path, data: function schema, name: function name
        log_data[KEY_DATA] = json.dumps(data)
        log_data[KEY_NAME] = name
        log_data[KEY_DATA_TYPE] = DATA_TYPE_MESSAGES
        log_data[KEY_TIMESTAMP] = get_current_timestamp_milli()
        log_data[KEY_WORKFLOW_ID] = get_workflow_id()
        ## required fields
        log_data[KEY_PROCESS_ID] = kwargs[KEY_PROCESS_ID] if KEY_PROCESS_ID in kwargs else ""
        log_data[KEY_AGENT_NAME] = kwargs[KEY_AGENT_NAME] if KEY_AGENT_NAME in kwargs else ""
        
        log_data.update(kwargs)
        log_data_list.append(log_data)
        ## add log to writer
        [writer.add_log(log_data) for log_data in log_data_list]
        writer.write_log()
    except Exception as e:
        print (e)
        s = traceback.format_exc()
        print (s)


def agent_loop(name, data, agent_name, process_id, **kwargs):
    """ AgentBoard 
        print an agent loop consists of usered defined stages and stage input, output and duration, e.g. Plan stage
        |agent_name|process_id|name|data|data_type|
        
        Required Fields: 
            name
            data
            agent_name
        
        Optional Fields:
            workflow_type: e.g. process/decision/start/end

        ## PLAN STAGE
        |agent 1|PLAN|PLAN START|{"args1": "1", "args2": "2"}|agent_loop|
        |agent 1|PLAN|PLAN Execution|{"duration": "30"|agent_loop|
        |agent 1|PLAN|PLAN END|{"result": "2"|agent_loop|


    """
    ## write logs
    writer = context_file_writer.get("context_file_writer")
    try:
        if not (type(data) == dict or isinstance(data, str)):
            print ("ERROR: agent_loop input data should be in python dict or str format")
            return
        log_data = {}

        data_clean = json.dumps(data) if type(data) == dict else data
        # require field
        log_data[KEY_NAME] = name
        log_data[KEY_DATA] = data_clean
        log_data[KEY_AGENT_NAME] = agent_name
        log_data[KEY_PROCESS_ID] = process_id        
        log_data[KEY_DATA_TYPE] = DATA_TYPE_AGENT_LOOP
        log_data[KEY_TIMESTAMP] = get_current_timestamp_milli()
        log_data[KEY_WORKFLOW_ID] = get_workflow_id()
        # Optional field
        log_data[KEY_WORKFLOW_TYPE] = kwargs[KEY_WORKFLOW_TYPE] if KEY_WORKFLOW_TYPE in kwargs else ""

        ## default values
        log_data.update(kwargs)
        if writer is not None:
            writer.add_log(log_data)
            writer.write_log()
        else:
            print("DEBUG: FileWriter is missing in contextvars...")
    except Exception as e:
        print (e)
        s = traceback.format_exc()
        print (s)

def main():

    import torch

    def calling_bing_tools(keyword:str, limit:int) -> str:
        url="https://www.bing.com/search?q=%s&limit=%d" % (keyword, limit)
        return url

    with FileWriter(logdir="./log", static="./static") as writer:
        print ("This is log file path %s" % writer.get_log_file_name())
        
        ## Text
        print ("#### DEBUG: Exporting Text Logs #### ")
        text(name="Plan Start Prompt", data="Please do image search with user input", agent_name="agent 1", process_id="plan")

        ## Dict
        print ("#### DEBUG: Exporting Dict Logs #### ")
        dict(name="Plan Input Args Dict", data=[{"arg1": 1, "arg2": 2}], agent_name="agent 1", process_id="plan")

        ## Image
        print ("#### DEBUG: Exporting Image Logs #### ")
        input_image = torch.mul(torch.rand(8, 3, 400, 600), 255).to(torch.int64)
        image(name="Plan Input Image", data=input_image, agent_name="agent 1", process_id="plan")

        ### Audio
        print ("#### DEBUG: Exporting Audio Logs #### ")        
        sample_rate = 16000  # 16 kHz
        duration_seconds = 2  # 2 seconds
        frequency = 440.0  # 440 Hz (A4 note)
        t = torch.linspace(0, duration_seconds, int(sample_rate * duration_seconds), dtype=torch.float32)
        waveform = (0.5 * torch.sin(2 * math.pi * frequency * t)).unsqueeze(0)  # Add channel dimension
        waveform = torch.unsqueeze(waveform, dim=0)
        audio(name="Plan Input Audio", data=waveform, agent_name="agent 1", process_id="plan")


        ## Video
        print ("#### DEBUG: Exporting Video Logs #### ")                
        T, H, W, C = 30, 64, 64, 3  # 30 frames, 64x64 resolution, 3 color channels
        video_tensor = torch.randint(0, 256, (T, H, W, C), dtype=torch.uint8)
        # Specify output file and frame rate
        frame_rate = 24  # Frames per second
        # Write the video to file
        video(name="Act Output Video", data=video_tensor, agent_name="agent 2", process_id="act")


        ## Tools or function calls
        print ("#### DEBUG: Exporting Tools Logs #### ")                        
        tool(name="Act RAG Tool Bing", data=[calling_bing_tools], agent_name="agent 2", process_id="act")

        ## Messages
        print ("#### DEBUG: Exporting Messages Logs #### ")  
        chat_messages = [{"role":"user", "content": "hello"}, {"role":"assistant", "content": "Hola! My name is bot."}, {"role":"user", "content": "Please help me summarize the stock market news."}]
        messages(name="Chatbot Messages", data=chat_messages, agent_name="chatbot", process_id="chat")

        # ## Act 
        # text(name="Act Start", data="Act Stage starts", agent_name="agent 1", process_id="act")

        # tool(name="Act Calling Bing Tools", data=[calling_bing_tools], agent_name="agent 1", process_id="act")
        # ## input
        # json(name="Act Input Args", data=[{"arg1": 1, "arg2": 2}], agent_name="agent 1", process_id="act")
        # ## output
        # json(name="Act Output Args", data=[{"result1": 1, "result2": 2}], agent_name="agent 1", process_id="act")


if __name__ == '__main__':
    main()

