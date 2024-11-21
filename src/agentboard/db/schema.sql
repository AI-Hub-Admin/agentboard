DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;
DROP TABLE IF EXISTS user_actions;

CREATE TABLE user_profile (
  id TEXT PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  avatar TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  ext_info TEXT
);


CREATE TABLE agent_activity (
  activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
  gmt_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  log_time    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  agent_id    TEXT NOT NULL,
  agent_name  TEXT NOT NULL,
  role        TEXT NOT NULL,
  content     TEXT NOT NULL,
  status      INTEGER NOT NULL,
  ext_info    TEXT NOT NULL,
  FOREIGN KEY (agent_id) REFERENCES user_profile (id)
);



--  Comment and Replies, comment on the post
CREATE TABLE comment (
  comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
  to_id  INTEGER,
  log_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  user_id    TEXT,
  content    TEXT,
  ext_info   TEXT,
  FOREIGN KEY (user_id) REFERENCES user_profile (id)  
);


CREATE TABLE reply (
  reply_id INTEGER PRIMARY KEY AUTOINCREMENT,
  to_id    INTEGER,
  log_time  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  user_id    TEXT,
  content    TEXT,
  ext_info   TEXT,
  FOREIGN KEY (user_id) REFERENCES user_profile (id)  
);

