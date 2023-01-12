##Not sure if I need this, since I'm not deploying to heroku

#This file is a requirement from heroku when deploying an app to heroku
#it specifies the commands to run our application
#uvicorn is the web server
#and we are specifying the location of our application
#app is the main directory, main is the main.py inside app and then app is the FastAPI instance in main.py
#so we are basically pointing heroku to where to find out main app

#heroku dynamically assigns port so we cant specify port here

web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
