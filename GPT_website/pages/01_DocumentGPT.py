import time
import streamlit as st

st.title("DocumentGPT")

##Checking if there's at least one
#Only Executed at first time, and then at second time, 
#it doesn't since some "messages" are saved in session_state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

###remembering the message using session_state
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})
    
#if the message in not yet recorded
for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False,)


##Functions
#Function that Ques the status widget
def status_in_que():
    with st.status("Embedding file...", expanded=True) as status:
        time.sleep(2)
        st.write("Getting the file")
        time.sleep(2)
        st.write("Embedding the file")
        time.sleep(2)
        st.write("Caching the file")
        status.update(label="Error", state="error")


##Creating user input and starting the conversation
message = st.chat_input("Send a message to the ai")

#checking if the message is not NULL(empty)
if message:
    if (message =="/reset"):
        st.session_state["messages"] = []
        send_message("Data reset completed", "ai", save=False)
    else:
        send_message(message, "human")
        time.sleep(2)
        send_message(f"You said: {message}", "ai")

        with st.sidebar:
            st.write(st.session_state)
    

#It seems that due to continuous rerunning there's no time to show up on the debug console
print(st.session_state)





