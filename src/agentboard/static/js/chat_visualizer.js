const chatbotToggler = document.querySelector(".chatbot-toggler");

const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");

let userMessage = null; 
const API_KEY = "API-KEY";
const inputInitHeight = chatInput.scrollHeight;

const createChatLi = (message, className) => {
    // Create a chat <li> element with passed message and className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi; // return chat <li> element
}


const createChatLiWithIcon = (message, className, userName, iconUrl) => {
    // Create a chat <li> element with passed message and className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    // let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    let iconHtml = "";
    if (iconUrl == null) {
        var userNameShort = "";
        if (userName.length >= 2) {
            userNameShort = userName.charAt(0) + userName.charAt(1);
        } else if (userName.length == 1) {
            userNameShort = userName.charAt(0);
        } else {
        }
        iconHtml = '<div class="div_icon_default_name_small"><a>' + userNameShort + '</a></div>';
    } else {
        iconHtml = '<div class="display_card_image_thumbnail_flex"><img class="display_card_image_thumbnail_img" src="'+ iconUrl + '"></div>';
    }
    if (className == "outgoing") {
        // let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
        var chatContent = `<p></p>`;
        chatLi.innerHTML = chatContent + iconHtml;
        chatLi.querySelector("p").textContent = message;
        return chatLi; // return chat <li> element
    } else {
        var chatContent = `<p></p>`;
        chatLi.innerHTML = iconHtml + chatContent;
        chatLi.querySelector("p").textContent = message;
        return chatLi; // return chat <li> element
    }
}

const createChatLiWithIconAndName = (message, className, userName, iconUrl) => {
    // Create a chat <li> element with passed message and className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    // let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    let iconHtml = "";
    if (iconUrl == null) {
        var userNameShort = "";
        if (userName.length >= 2) {
            userNameShort = userName.charAt(0) + userName.charAt(1);
        } else if (userName.length == 1) {
            userNameShort = userName.charAt(0);
        } else {
        }
        iconHtml = '<div class="div_icon_default_name_small"><a>' + userNameShort + '</a></div>';
    } else {
        iconHtml = '<div class="display_card_image_thumbnail_flex"><img class="display_card_image_thumbnail_img" src="'+ iconUrl + '"></div>';
    }
    if (className == "outgoing") {
        // let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;

        var userNameContent = '<a class="chat_username_right">' + userName + '</a>'
        var chatContent = '<div class="chat_username_message">' + userNameContent + '<p></p>' + '</div>';
        chatLi.innerHTML = chatContent + iconHtml;
        chatLi.querySelector("p").textContent = message;
        return chatLi; // return chat <li> element
    } else {
        // var chatContent = `<p></p>`;
        var userNameContent = '<a class="chat_username_left">' + userName + '</a>'        
        var chatContent = '<div class="chat_username_message">' + userNameContent + '<p></p>' + '</div>';
        chatLi.innerHTML = iconHtml + chatContent;
        chatLi.querySelector("p").textContent = message;
        return chatLi; // return chat <li> element
    }
}


const generateResponsePlain = (chatElement) => {
    const API_URL = "https://api.openai.com/v1/chat/completions";
    const messageElement = chatElement.querySelector("p");
    messageElement.textContent = "this is agentboard"
    // Define the properties and message for the API request
    // const requestOptions = {
    //     method: "POST",
    //     headers: {
    //         "Content-Type": "application/json",
    //         "Authorization": `Bearer ${API_KEY}`
    //     },
    //     body: JSON.stringify({
    //         model: "gpt-3.5-turbo",
    //         messages: [{role: "user", content: userMessage}],
    //     })
    // }

    // // Send POST request to API, get response and set the reponse as paragraph text
    // fetch(API_URL, requestOptions).then(res => res.json()).then(data => {
    //     messageElement.textContent = data.choices[0].message.content.trim();
    // }).catch(() => {
    //     messageElement.classList.add("error");
    //     messageElement.textContent = "Oops! Something went wrong. Please try again.";
    // }).finally(() => chatbox.scrollTo(0, chatbox.scrollHeight));
}


const generateResponse = (chatElement) => {
    const API_URL = "https://api.openai.com/v1/chat/completions";
    const messageElement = chatElement.querySelector("p");

    // Define the properties and message for the API request
    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${API_KEY}`
        },
        body: JSON.stringify({
            model: "gpt-3.5-turbo",
            messages: [{role: "user", content: userMessage}],
        })
    }

    // Send POST request to API, get response and set the reponse as paragraph text
    fetch(API_URL, requestOptions).then(res => res.json()).then(data => {
        messageElement.textContent = data.choices[0].message.content.trim();
    }).catch(() => {
        messageElement.classList.add("error");
        messageElement.textContent = "Oops! Something went wrong. Please try again.";
    }).finally(() => chatbox.scrollTo(0, chatbox.scrollHeight));
}

const handleChat = () => {
    userMessage = chatInput.value.trim(); // Get user entered message and remove extra whitespace
    if(!userMessage) return;

    // Clear the input textarea and set its height to default
    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;

    // Append the user's message to the chatbox
    chatbox.appendChild(createChatLi(userMessage, "outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);
    
    setTimeout(() => {
        // Display "Thinking..." message while waiting for the response
        const incomingChatLi = createChatLi("Thinking...", "incoming");
        chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0, chatbox.scrollHeight);
        generateResponsePlain(incomingChatLi);
    }, 600);
}

chatInput.addEventListener("input", () => {
    // Adjust the height of the input textarea based on its content
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
    // If Enter key is pressed without Shift key and the window 
    // width is greater than 800px, handle the chat
    if(e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
        e.preventDefault();
        handleChat();
    }
});

sendChatBtn.addEventListener("click", handleChat);
closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));




// display dialogue 

        function displayDialogueFromJson(jsonStr) {
          try {
            if (jsonStr == null) {
              return;
            }

            // get template parameters
            var templateElem = document.querySelector('input[name="template_radio"]:checked')
            var template = "";
            if (templateElem != null) {
              template = templateElem.value;
            } else {
              template = "default";
            }
            // Change all Background
            var chatbot = document.querySelector('#div_chatbot_id');
            if (template == "wechat") {
              chatbot.className = "chatbot_wechat";
            } else {
              chatbot.className = "chatbot";
            }
            chatbox.innerHTML = "";


            var jsonObj = JSON.parse(jsonStr);
            // check required keys
            var keyMessage = "messages";
            var keyAgents = "agent";
            var keyAvatars = "avatar";
            var keyDialogueTitle = "dialogue_title";
            var keyDialogueBackgroundImageUrl = "dialogue_background_url";

            // incoming roles
            // outcoming roles
            var incomingRolesList = ["ai", "assistant", "system", "AI"];
            var outgoingRolesList = ["human", "user"];


            var messages = null;
            if (jsonObj[keyMessage] == null) {
                messages = jsonObj;
            } else {
                messages = jsonObj[keyMessage];
            }

            var agent = jsonObj[keyAgents];

            for (var i = 0; i < messages.length; i++) {
              var msg = messages[i];
              var id = msg["id"];
              var timestamp = msg["timestamp"];
              var content = msg["content"];
              var role = msg["role"];
              // update id for display
              if (id == null && role != null) {
                id = role
              }

              // agent obj
              var agentAvatarURL = null;
              if (agent != null) {
                var agentObj = agent[id];
                if (agentObj != null && agentObj[keyAvatars] != null) {
                  agentAvatarURL = agentObj[keyAvatars];
                }
              }
              if (outgoingRolesList.indexOf(role) > -1) {
                // var chatoutgoing = createChatLi(content, "outgoing");
                var chatoutgoing = "";
                if (template == "wechat") {
                  chatoutgoing = createChatLiWithIconAndName(content, "outgoing", id, agentAvatarURL);
                } else {
                  chatoutgoing = createChatLiWithIcon(content, "outgoing", id, agentAvatarURL);
                }

                chatbox.appendChild(chatoutgoing);
                chatbox.scrollTo(0, chatbox.scrollHeight);
              } else if (incomingRolesList.indexOf(role) > -1) {
                // var chatincoming = createChatLi(content, "incoming");
                var chatincoming = "";

                if (template == "wechat") {
                  chatincoming = createChatLiWithIconAndName(content, "incoming", id, agentAvatarURL);
                } else {
                  chatincoming = createChatLiWithIcon(content, "incoming", id, agentAvatarURL);
                }

                chatbox.appendChild(chatincoming);
                chatbox.scrollTo(0, chatbox.scrollHeight);
              } else {
                var chatincoming = createChatLiWithIcon(content, "incoming", id, agentAvatarURL);
                chatbox.appendChild(chatincoming);
                chatbox.scrollTo(0, chatbox.scrollHeight);
              }
            }

            // set dialogue attributes
            var dialogueTitleValue = jsonObj[keyDialogueTitle];
            if (dialogueTitleValue != null) {
              var dialogueTitleElem = document.getElementById("div_chatbot_title_a");
              dialogueTitleElem.innerText = dialogueTitleValue;
            }

            var dialogueImageUrlValue = jsonObj[keyDialogueBackgroundImageUrl];
            if (dialogueImageUrlValue != null) {
              const chatboxContentWrapper = document.querySelector(".div_chatbox_content_wrapper");
              var backgroundImageValue = 'url("' + dialogueImageUrlValue + '")';
              chatboxContentWrapper.style.backgroundImage = backgroundImageValue;
            }

            // console.log(jsonObj);
          } catch (e) {
            console.error('Failed to convert json...', e);
            alert("Input Json is Invalid!");
          }
        }

