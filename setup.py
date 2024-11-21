from setuptools import setup, find_packages
import pathlib

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    "Flask>=2.0.0",
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "torchvision>=0.17.0",
    "tensorflow>=1.13.0",
    "pillow>=9.0.0",
    "contextvars>=2.4"
]

extras = dict()
extras['dev'] = []
extras['test'] = extras['dev'] + []

setup(
    name="agentboard",  
    version="0.0.2",    
    description="AI Agent Visualization Toolkit, AI Agents development, such as agent loop, chat messages, tools/functions, workflow and raw data types including text, dict or json, image, audio, video, etc. ",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author_email="aihubadmin@126.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="Agent Board Visualization,AI Agent,Visualize",
    packages=find_packages(where="src"),
    install_requires=install_requires,
    include_package_data=True,
    package_dir={"": "src"},
    package_data={
        'agentboard.static.video': ['*.mp4'],        
        'agentboard.static.audio': ['*.wav'],        
        'agentboard.static.css': ['*.css'],        
        'agentboard.static.js': ['*.js'],        
        'agentboard.static.img': ['*.png', '*.jpg', '*.jpeg', '*.webp'],        
        'agentboard.templates': ['*.html'],
        'agentboard.log': ['*.log'],
        'agentboard.db': ['*.sql', '*.db', '*.sh']
    },
    python_requires=">=3.6",
    extras_require=extras,
    project_urls={
        "homepage": "http://www.deepnlp.org/blog?category=agentboard",
        "repository": "https://github.com/AI-Hub-Admin/agentboard"
    },
    entry_points={
        # CLI command to start the app
        "console_scripts": [
            "agentboard=agentboard.run_agent_board:run_command_line"
        ]
    }  
)
