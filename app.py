from apikey import apikey
import os 
import json

os.environ['OPENAI_API_KEY'] = apikey

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

from flask import Flask, request, jsonify

app = Flask(__name__)



prompt_template = PromptTemplate(
    input_variables = ['text', 'parameters', 'table'], 
    template='parse only these parameters from the text and make clear json with values by template from {table} in response with it PARAMETERS: {parameters} TEXT: {text}'
)

llm = OpenAI(model_name="gpt-3.5-turbo-0301") 

# Memory 
text_memory = ConversationBufferMemory(input_key='text', memory_key='chat_history')


car_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, memory=text_memory)

parameters = 'make, model, year, mileage in kilometers, steering side, engine volume in cm3 with only number, fuel type, exporter country, transmission type, color, interior color'

@app.route('/', methods=['GET'])
def get_car():
    # query_parameters = request.args

    # car_details = query_parameters.get('car_details')

    # data = json.loads(unquote(request.query_string.partition('&')[0]))
    # car_details = data.get('car_details')
    car_details = request.json.get('car_details', None) 
    price = request.json.get('price', None) 
    print(car_details)


    res_json = json.loads(car_chain.run(text=car_details, parameters=parameters))

    print(res_json)

    res_json.update({"price": price})

    if str(res_json['model'])[0].isdigit() and res_json['make'] == "BMW":
        series = str(res_json['model'])[0]
        res_json['model'] = series + "-series"

    return res_json


if __name__ == "__main__":
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=False, threaded=True, port=5000)