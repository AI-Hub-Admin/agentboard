# -*- coding: utf-8 -*-
# @Time    : 2024/06/27

import agentboard as ab


def main():


    with ab.summary.FileWriter(logdir="./log", static="./static") as writer:
        print ("This is log file path %s" % writer.get_log_file_name())
        
        ## Text
        print ("#### DEBUG: Exporting Text Logs #### ")
        ab.summary.text(name="Plan Start Prompt", data="Please do image search with user input", agent_name="agent 1", process_id="plan")

        ## Dict
        print ("#### DEBUG: Exporting Dict Logs #### ")
        ab.summary.dict(name="Plan Input Args Dict", data=[{"arg1": 1, "arg2": 2}], agent_name="agent 1", process_id="plan")

        ## Image
        print ("#### DEBUG: Exporting Image Logs #### ")
        input_image = torch.mul(torch.rand(8, 3, 400, 600), 255).to(torch.int64)
        ab.summary.image(name="Plan Input Image", data=input_image, agent_name="agent 1", process_id="plan")

        ### Audio
        print ("#### DEBUG: Exporting Audio Logs #### ")        
        sample_rate = 16000  # 16 kHz
        duration_seconds = 2  # 2 seconds
        frequency = 440.0  # 440 Hz (A4 note)
        t = torch.linspace(0, duration_seconds, int(sample_rate * duration_seconds), dtype=torch.float32)
        waveform = (0.5 * torch.sin(2 * math.pi * frequency * t)).unsqueeze(0)  # Add channel dimension
        waveform = torch.unsqueeze(waveform, dim=0)
        ab.summary.audio(name="Plan Input Audio", data=waveform, agent_name="agent 1", process_id="plan")

        ## Video
        print ("#### DEBUG: Exporting Video Logs #### ")                
        T, H, W, C = 30, 64, 64, 3  # 30 frames, 64x64 resolution, 3 color channels
        video_tensor = torch.randint(0, 256, (T, H, W, C), dtype=torch.uint8)
        # Specify output file and frame rate
        frame_rate = 24  # Frames per second
        # Write the video to file
        ab.summary.video(name="Act Output Video", data=video_tensor, agent_name="agent 2", process_id="act")

if __name__ == '__main__':
    main()
