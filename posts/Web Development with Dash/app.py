#!/usr/bin/env python
# coding: utf-8

# # Building a Message Bank Web App with Dash by Plotly
# In this tutorial, we'll walk through the process of building a simple web application using Dash by Plotly. The app we are creating is a message bank, which allows users to submit messages and view a sample of messages stored in a database. By the end of this tutorial, you will have a functional web app that demonstrates the use of Dash callbacks, database management, and basic styling.

# ## Building the Web App using Dash
# First, let's build the dash web app before adding functionalities.
# 
# We’ll start by importing the required libraries and initializing the app. Then, we will create a message bank to store users' daily mood quotes:

# In[3]:


from dash import Dash, html, dcc, Input, Output, State
app = Dash(__name__)
app.title = "Message Bank: Daily Mood Quotes"


# Next, we will create the app layout. It consists of two sections: submit and view.
# 
# The submission functionality will have three user interface elements:
# - a text box for submitting a message
# - a text box for submitting the name of the user
# - a "submit" button
# 
# The view functionality will have two user interface elements:
# - a section for displaying random messages
# - an "update" button
# 
# Let’s also apply some styling to the layout, including fonts, colors, margins, and more.

# In[5]:


app.layout = html.Div([
    html.H1("Message Bank: Daily Mood Quotes", 
            style={
                "textAlign": "center",
                "color": "#084a8c",
                "fontFamily": "'Roboto', sans-serif",
                "marginBottom": "20px"
    }),
    
    # Submission Section
    html.H2("Submit", style={
        "color": "#7848a8",
        "fontFamily": "'Roboto', sans-serif"
    }),
    html.Label("Your message:", style={
        "fontFamily": "'Roboto', sans-serif",
        "marginTop": "10px",
        "marginBottom": "10px"
        }), 
    html.Br(),
    dcc.Textarea(id="message", style = {
        "width": "50%",
        "height": "100px",
        "verticalAlign": "top"
    }),
    html.Br(),
    html.Label("Your Name or Handle:", style={
        "fontFamily": "'Roboto', sans-serif",
        "marginTop": "10px",
        "marginBottom": "10px"
        }),
    html.Br(),
    dcc.Input(id="handle", type="text"),
    html.Br(),
    html.Button("Submit", id="submit-button",
                style={
        "backgroundColor": "#e0d1f0",
        "color": "white",
        "padding": "10px 20px",
        "border": "none",
        "borderRadius": "25px",
        "cursor": "pointer",
        "fontFamily": "'Roboto', sans-serif",
        "fontSize": "16px",
        "marginTop": "10px",
        "marginBottom": "10px"
    }),
    html.Div(id="submit-confirmation", style={"fontFamily": "'Roboto', sans-serif"}),
    
    # View Messages Section
    html.Hr(),
    html.H2("View", style={
        "color": "#7848a8",
        "fontFamily": "'Roboto', sans-serif"
    }),
    html.Div(id="message-display",
             style = {"fontFamily": "'Roboto', sans-serif"}),
    html.Button("Update", id="update-button", style={
        "backgroundColor": "#e0d1f0",
        "color": "white",
        "padding": "10px 20px",
        "border": "none",
        "borderRadius": "25px",
        "cursor": "pointer",
        "fontFamily": "'Roboto', sans-serif",
        "fontSize": "16px",
        "marginTop": "10px",
        "marginBottom": "10px"
    }),  
])


# ## Creating the Messages Database
# In this section, we will create a database to store the messages.
# 
# For database management in the app, we will create two python functions. The first function is `get_message_db()`. It handles the creation of the database and the messages table if they don’t already exist.

# In[7]:


import sqlite3

message_db = None

def get_message_db():
    """
    create or connect to the sqlite database and ensure the messages table exists
    retun: the database connection
    """
    global message_db
    if message_db:
        return message_db
    
    else:
        # connect to database (or create it if it doesn't exist)
        message_db = sqlite3.connect("messages_db.sqlite", check_same_thread=False)

        # create the messages table if it doesn't exist
        cmd = """
            CREATE TABLE IF NOT EXISTS messages(
                handle TEXT,
                message TEXT
            )
            """
        
        cursor = message_db.cursor()
        cursor.execute(cmd)
        return message_db


# The function above first check whether there is a database called `message_db` defined in the global scope. If not, it will connect to that database and assign it to the global variable `message_db`. 
# 
# It will also check whether a table called `messages` exists in the database, and create it if not. Using SQL command, we create a table with two columns: handle and message. At the end, the function returns the connection `message_db`.

# The second function is `insert_message(handle, message)`, which inserts the message into the database.

# In[10]:


def insert_message(handle, message):
    """
    insert a new message into the messages database
    """
    db = get_message_db()
    cursor = db.cursor()
    
    cmd = "INSERT INTO messages (handle, message) VALUES (?, ?)"
    cursor.execute(cmd, (handle, message))
    
    db.commit() # save the row insertion
    global message_db
    message_db = None # set the global message_db to none
    db.close() # close the database connection


# The function above uses a cursor to insert the message into the message database. We use parameterized query with ? as placeholders, which properly handles the string quotes.
# 
# At the end of the function, we run `db.commit()` after row insertion to ensure the change has been saved, and we will also close the database connection. Note that a column called rowid is automatically generated by default, which gives an integer index to each row we add to the database.

# ## Enable Submissions through Callback Function
# 
# Now that we have created functions for database management, we will add a submit functionality to our web app. 
# 
# To enable interaction with a input component and change the resulting output component, we will need to create a callback function `submit()` to update the components. 

# In[13]:


@app.callback(
    Output("submit-confirmation", "children"), # the output will update this div element
    Input("submit-button", "n_clicks"),
    State("handle", "value"),
    State("message", "value"),
    prevent_initial_call=True
)
def submit(n_clicks, handle, message):
    """
    insert the submitted message into the database and display a confirmation line
    """
    if handle and message:
        insert_message(handle, message)
        return "Thank you for your submission!" # successful submission
    else:
        return "Error: please check if any field is empty!" # error occurs


# This function inserts the submitted message into the messages database and display a confirmation line if submission is successful, and prints an error message if it failed.
# 
# The callback specifies each specific input and output components, as well as their specific properties. For example, the output will be the "submit-confirmation" component and we will update its children html element.

# ## Enabling Viewing through Callback Function
# After enabling the submit functionality, we will add the second component, viewing submissions. In this section, we will write two other functions to enable message display.
# 
# First, we will write a function called `random_messages(n)`, which will return a collection of n random messages from the message_db, or fewer if necessary. We have set a default number of 5 messages to be displayed.

# In[16]:


def random_messages(n=5):
    """
    return a collection of n random messages from the database
    or fewer if necessary
    """

    db = get_message_db()
    cursor = db.cursor()

    cmd = "SELECT handle, message FROM messages ORDER BY RANDOM() LIMIT ?" # parameterized query
    cursor.execute(cmd, (n,))
    messages = cursor.fetchall()

    db.close()
    global message_db
    message_db = None # set the global message_db to none
    return messages


# Next, we will write a callback function called  `view()` to display random messages. These messages will be displayed in the section with the component id "message-display".

# In[18]:


@app.callback(
        Output("message-display", "children"),
        Input("update-button", "n_clicks"),
        prevent_initial_call=True
)
def view(n_clicks):
    """
    display random messages
    """
    messages = random_messages()

    # iterate through each of the message
    messages_list = []
    for handle, message in messages:
        messages_list.append(html.P([
            message,
            html.Br(),
            "- " + handle
        ]))

    return messages_list


# This function first calls  `random_messages()` to grab some messages and display them using a loop. This function is triggered when the "update" button is pressed, as specified in the callback.

# ## Running the Web App
# Now that we've set up all the features and layout of our website, let's run the app and test its functionality to ensure everything works as expected.

# In[21]:


if __name__ == "__main__":
    app.run_server(port=8050, debug=True)


# First, let’s test the submission function. It works perfectly, displaying a thank you message after a successful submission.
# ![screenshot of submit function](submit.png)

# Next, let's check the message display feature. We’ve submitted seven messages, each labeled with keywords such as "one", "two", "three", and so on, to indicate the order of submission. Since we’ve capped the display to a maximum of five messages, clicking the update button will randomly select five messages from the database.
# 
# 
# ![Screnshot of Message Display](update.png)

# Great! We see that the web app is working as desired. With these functions tested and confirmed, our web app is now ready for deployment! :)
