import os
import time
from loguru import logger
import json
from datetime import datetime
import math
import operator
from openai import OpenAI
from prompts import PROMPTS
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()
MY_DEEPSEEK_KEY=os.getenv("MY_DEEPSEEK_KEY")
import sqlite3 as sq
from datetime import datetime


def db_start():
    global db, cur 

    db = sq.connect('response.db')
    cur = db.cursor()

    cur.execute("CREATE TABLE IF NOT EXISTS messages_v2(session_id INT, user TEXT, step TEXT, agent TEXT,  response TEXT, reasoning TEXT, created_at TIMESTAMP)")

    db.commit()


def add_to_history(session_id, user, step, agent, response, reasoning):
    now = datetime.now()
    time = now.strftime("%d/%m/%y %H:%M:%S")
    cur.execute("INSERT INTO messages_v2 VALUES(?, ?, ?, ?, ?, ?, ?)", (session_id, user, step, agent, response,reasoning, time))
    db.commit()





def main(session_id: int, user: str, text_to_translate: str):
    db_start()
    client = OpenAI(
    base_url="https://api.artemox.com/v1",
    api_key=MY_DEEPSEEK_KEY,
    )

    class State(TypedDict):
        session_id: int
        user: str #Имя пользователя для сохранения в бд
        phase: str #Состояние графа, всегда принимает только одно значение | Нулевое состояние: "stage_0_translation"
        text_to_translate: str #Текст для перевода
        translation: Annotated[Dict[str, str], operator.or_] #Текст перевода
        translation_reasoning: Annotated[Dict[str, str], operator.or_] #Текст перевода
        places_to_visit: str #что посетить в России и в Китае


    def save_state(state: State, agent: str):
        """Сохраняет состояние в JSON файл"""
        # Конвертируем в обычный dict
        state_dict = dict(state)
        
        # Добавляем timestamp для отслеживания
        state_with_meta = {
            "timestamp": datetime.now().isoformat(),
            "phase": state["phase"],
            "state": state_dict
        }

        filename = f"system_states\\{agent}_phase_{state['phase']}_{datetime.now().isoformat().replace(':', '-')}.json"
        
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(state_with_meta, ensure_ascii=False) + '\n')

    class BaseExpert:
        """Базовый класс для всех экспертов"""
        def __init__(self, role_name: str, description: str) -> None:
            self.prompt = PROMPTS[role_name]
            self.role_name = role_name
            self.__doc__ = description
        

        def __call__(self, state: State):
            
            #-------------------- Шаг 1 --------------------

            if state['phase'] =='translation':
                save_state(state, self.role_name)
                logger.info(f'phase: {state['phase']} \n agent: {self.role_name}')
                completion = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                    "role": "system",
                    "content": self.prompt
                    },
                    {
                    "role": "user",
                    "content": state['text_to_translate']
                    }
                ]
                )
                
                logger.info(f"state: \n{state['phase']} \n agnet: \n{self.role_name} \n response: {completion.choices[0].message.content} \n resonign: {completion.choices[0].message.reasoning_content}")
                info_message = f"""DATE: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\nstate: \n{state['phase']} \n agnet: \n{self.role_name} \n response: {completion.choices[0].message.content} \n resonign: {completion.choices[0].message.reasoning_content}"""

                with open(f"logs\\{state['phase']}.txt", 'a', encoding='utf-8') as f:
                    f.write(info_message)

                return {'translation': {self.role_name: completion.choices[0].message.content},
                        "translation_reasoning":{self.role_name: completion.choices[0].message.reasoning_content} }
            
            #-------------------- ШАГ 2 --------------------


            if state['phase'] == 'popular_places':

                save_state(state, self.role_name)
                logger.info(f'phase: {state['phase']} \n agent: {self.role_name}')
                completion = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[

                    {
                    "role": "user",
                    "content": self.prompt
                    }
                ]
                )


                add_to_history(session_id=state['session_id'],
                            user = state['user'],
                            step = state['phase'],
                            agent = self.role_name,
                            response = completion.choices[0].message.content,
                            reasoning=completion.choices[0].message.reasoning_content)
                logger.info(f'state: {state['phase']}\n agent: {self.role_name} \n DB save')
                
                logger.info(f"state: \n{state['phase']} \n agnet: \n{self.role_name} \n response: {completion.choices[0].message.content} \n resonign: {completion.choices[0].message.reasoning_content}")
                info_message = f"""DATE: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\nstate: \n{state['phase']} \n agnet: \n{self.role_name} \n response: {completion.choices[0].message.content} \n resonign: {completion.choices[0].message.reasoning_content}"""

                with open(f"logs\\{state['phase']}.txt", 'a', encoding='utf-8') as f:
                    f.write(info_message)

                return {"places_to_visit":completion.choices[0].message.content}


    english_translator = BaseExpert(role_name="english_translator",
                            description=" ")
    chinese_translotor = BaseExpert(role_name="chinese_translotor",
                            description=" ")
    traveler = BaseExpert(role_name="traveler",
                            description=" ")

    def switch_states(state: State):
        """Меняет состояние графа"""
            
        if state['phase'] == 'translation':
            print('меняю translation на popular_places')
            return {"phase":"popular_places"}
        
        if state['phase'] == 'popular_places':
            print('меняю popular_places на end')
            return {"phase":"end"}
        

    #Задает дальнейший шаг
    def current_state(state: State):
        return state['phase']


    def empty_fucntion(state: State):
        """Останавливает паралелльную работу агентов"""
        print(f'empty, state = {state['phase']}')
        pass



    def agents_translators(state:State):
        """Возвращает список всех активных агентов"""

        return ["english_translator", "chinese_translotor"]



    def agents_parallel_work_result_to_db(state: State):
        """Сохраняет результаты параллельной работы агентов в бд"""
        logger.info('Включилась функция agents_work_result_to_db')
        for agent in ["english_translator", "chinese_translotor"]:

            add_to_history(session_id=state['session_id'],
                        user = state['user'],
                        step = 'translation',
                        agent = agent,
                        response = state['translation'][agent],
                        reasoning=state['translation_reasoning'][agent])
        logger.info('Данные сохранены в базу')




    translators_builder = StateGraph(State)

    translators_builder.add_node("translators_results", empty_fucntion)
    translators_builder.add_node("english_translator", english_translator)
    translators_builder.add_node("chinese_translotor", chinese_translotor)
    translators_builder.add_node("agents_parallel_work_result_to_db", agents_parallel_work_result_to_db)
    translators_builder.add_conditional_edges(START,
                                                    agents_translators,
                                                    {
                                                        "english_translator": "english_translator",
                                                        "chinese_translotor": "chinese_translotor",
                                                    })

    translators_builder.add_edge("english_translator", "translators_results") 
    translators_builder.add_edge("chinese_translotor", "translators_results") 
    translators_builder.add_edge("translators_results", "agents_parallel_work_result_to_db") 


    translators = translators_builder.compile()

    global_block_graph_builder = StateGraph(State)

    global_block_graph_builder.add_node("translators", translators)
    global_block_graph_builder.add_node("router", empty_fucntion)
    global_block_graph_builder.add_node("switch_state", switch_states)
    global_block_graph_builder.add_node("traveler", traveler)



    global_block_graph_builder.add_edge(START, "translators") #state['phase'] == 'translation'
    global_block_graph_builder.add_edge("translators", "switch_state") #1. state['phase'] == 'stage_0_translation' -> state['phase'] == 'self_introduction'
    global_block_graph_builder.add_edge("switch_state", "router")
    global_block_graph_builder.add_conditional_edges("router",
                                                    current_state,
                                                    {
                                                    'popular_places':"traveler", #state['phase'] == 'popular_places'
                                                    'end': END #state['phase'] == 'end'
                                                    })


    global_block_graph_builder.add_edge("traveler", "switch_state") 





    global_block_graph = global_block_graph_builder.compile()


    for event in global_block_graph.stream({
            "session_id": session_id,
            "user" : user,
            "phase": "translation",
            "text_to_translate": text_to_translate,

    }, {"recursion_limit": 200}):
        # pprint(event, max_depth=10)
        pass

# if __name__ == "__main__":
#     main(session_id='2',
#          user='Mike',
#          text_to_translate='Какая погода сегодня?') 