## FLASK CONSTANTS

FLASK_PORT = 5000
DEFAULT_LOG_DIR = "./log"
DEFAULT_STATIC_DIR = "./static"

STATIC_FOLDER_PATH = "static"
TEMPLATE_FOLDER_PATH = "templates"

### REST Agent Constants
REST_POST_AGENT_ACTIVITIES = "/agent/activities"
DB_PATH_SQLLITE = "db/sqllite_database.db"
SCHEMA_SQL_PATH = "db/schema.sql"

### LOG FOLDER

LOG_FILE_EXT_LOG = "log"


KEY_LOG_TYPE = "log_type"
LOG_TYPE_PROCESS = "PROCESS"

KEY_NAME = "name"
KEY_DATA = "data"
KEY_DATA_TYPE = "data_type"
DATA_TYPE_TEXT = "text"
DATA_TYPE_DICT = "dict"
DATA_TYPE_IMAGE = "image"
DATA_TYPE_AUDIO = "audio"
DATA_TYPE_VIDEO = "video"
DATA_TYPE_TOOL = "tool"
DATA_TYPE_MESSAGES = "messages"
DATA_TYPE_AGENT_LOOP = "agent_loop"
DATA_TYPE_WORKFLOW = "workflow"

KEY_PROCESS_ID = "process_id"
KEY_AGENT_NAME = "agent_name"
KEY_TIMESTAMP  = "timestamp"
KEY_WORKFLOW_ID = "workflow_id"
KEY_WORKFLOW_TYPE = "workflow_type"

KEY_MESSAGE_ROLE = "role"
KEY_MESSAGE_CONTENT = "content"


WORKFLOW_TYPE_PROCESS = "process"
WORKFLOW_TYPE_DECISION = "decision"
WORKFLOW_TYPE_TEXT = "text"
WORKFLOW_TYPE_START = "start"
WORKFLOW_TYPE_END = "end"

## 
KEY_FILE_EXT = "file_ext"
DEFAULT_VIDEO_EXT = ".mp4"
DEFAULT_AUDIO_EXT = ".wav"
DEFAULT_IMAGE_EXT = ".png"
# AUDIO
KEY_AUDIO_SAMPLE_RATE = "sample_rate"
DEFAULT_AUDIO_SAMPLE_RATE = 16000

## VIDEO
KEY_FRAME_RATE = "frame_rate"
KEY_VIDEO_CODECS = "video_codec"
DEFAULT_FRAME_RATE = 24
DEFAULT_VIDEO_CODECS = "mpeg4"


KEY_CHART_TYPE = "chart_type"
CHART_TYPE_WORKFLOW = "workflow"
CHART_TYPE_SWITCH_TAB = "switch_tab"

