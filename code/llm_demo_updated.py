from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import re
import json
import pandas as pd


def fetch_user_data(user_id: int) -> Dict[str, str]:
    user_data = pd.read_csv(
        'ml-1m/users.dat', sep='::',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python', encoding='latin-1')
    user_info = user_data.loc[user_data["UserID"] == user_id]


    gender_info = user_info["Gender"].values[0]
    gender = 'Female' if gender_info == 'F' else 'Male' if gender_info == 'M' else '<UKN>'

    return {
        "gender": gender,
        "age": int(user_info["Age"].values[0]),
        "Occupation": int(user_info["Occupation"].values[0])
    }

def fetch_movie_data(movie_id: int) -> Dict[str, str]:
    movie_data = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python', encoding='latin-1', names=["MovieID", "Name", "Genres"])
    movie_info = movie_data[movie_data["MovieID"] == movie_id]


    genres = movie_info["Genres"].values[0].split('|')

    return {
        "title": movie_info["Name"].values[0],
        "genres": genres
    }

def fetch_data(user_id: int, movie_sequence: list) -> Dict[str, Dict]:
    return {
        'user': fetch_user_data(user_id),
        'movie': [fetch_movie_data(movie_id) for movie_id in movie_sequence]
    }


def main(user_query: str):

    entrypoint_agent_system_message = """
    """
    llm_config ={}# need to fill
    entrypoint_agent = ConversableAgent("entrypoint_agent",
                                        system_message=entrypoint_agent_system_message,
                                        llm_config=llm_config,
                                        human_input_mode='NEVER')
    entrypoint_agent.register_for_execution(name="fetch_data")(fetch_data)

    fetchdata_agent_system_message = "You are an AI assistant. Your role is to determine the appropriate parameters for the `fetch_data` function."
    fetchdata_agent = ConversableAgent("fetchdata_agent",
                                       system_message=fetchdata_agent_system_message,
                                       llm_config=llm_config,
                                       max_consecutive_auto_reply=1,
                                       human_input_mode='NEVER')
    fetchdata_agent.register_for_llm(name="fetch_data", description="Fetches the user information and movie information.")(fetch_data)



    analysis_agent_system_message = """
    You are an AI assistant tasked with analyzing a user's movie-watching preferences and demographic information. Output the analysis in JSON format, including the following fields:
    {
        "Demographic Analysis": "<Your analysis here>",
        "Genre Preference": "<Your analysis here>",
        "Year Preference": "<Your analysis here>",
        "Summary": "<Summary of the User's Movie-Watching Personality>"
    }
    Only output the JSON object.
    """

    rec_agent_system_message = """
    Using the user's preferences, recommend a list of movies that align with the user's tastes across multiple aspects. Follow these steps to create a well-rounded set of recommendations:
    - Genre-Based Recommendations: Recommend 20 movies that align with the user's genre preferences.
    - Year-Based Recommendations: Recommend 20 movies that fit the user's preferred era or release period.
    - Actor-Based Recommendations: Recommend 20 movies that feature the user's preferred actors.
    - Overall Profile-Based Recommendations: Recommend 20 movies that best align with the user's complete profile.
    After generating the lists, rank all recommended movies from highest to lowest based on their overall fit with the user's profile.
    Present the ranked output in a single Python list. An example format is shown below:
    ```json
    [
    "Movie Title 1",
    "Movie Title 2",
    ...
    ]
    """


    comment_simulator_system_message = """
    Assume you are the user who has watched each of the recommended movies. For each movie, write a personal and honest review that reflects your genuine thoughts and feelings, aiming to provide helpful feedback to others considering watching the movie.

    **When writing the comments:**

    - **Adopt the user's voice and perspective**, considering their preferences, personality traits, and demographic information.
    - **Be honest and critical**. Highlight both the strengths and weaknesses of the movie from the user's point of view.
    - **Provide specific examples or aspects** that stood out to you, whether positive or negative.
    - **Aim to help others** decide whether the movie is worth watching, based on your experience.

    **Structure your review to cover the following aspects:**

    1. **Plot and Storyline**: Was it engaging? Did it hold your interest? Were there any plot holes or predictable elements?
    2. **Characters and Acting**: How did the performances resonate with you? Were the characters well-developed and believable?
    3. **Visual Effects and Cinematography**: Did the visual elements enhance the movie? Were there any standout scenes?
    4. **Themes and Messages**: Did the underlying themes resonate with you? Were they presented effectively?
    5. **Personal Impact and Enjoyment**: How did the movie make you feel overall? Would you watch it again or recommend it to others?

    **Example of a realistic and critical comment:**

    ```json
    {
        "movie_title": "Example Movie",
        "comments": {
            "Plot and Storyline": "The storyline started strong but lost momentum halfway through. Some plot points felt underdeveloped.",
            "Characters and Acting": "The lead actor is OK, but supporting characters lacked depth.",
            "Visual Effects and Cinematography": "The cinematography was stunning, especially the scenes shot at sunset.",
            "Themes and Messages": "The movie touched on important themes of redemption, but didn't explore them fully.",
            "Personal Impact and Enjoyment": "Overall, I enjoyed parts of the movie but felt it didn't live up to its potential. I might not watch it again."
        }
    }
   
    
    **Instructions:**
    Provide the comments in the following JSON format, one per movie:

    {
        "movie_title": "<title>",
        "comments": {
            "Plot and Storyline": "<comment>",
            "Characters and Acting": "<comment>",
            "Visual Effects and Cinematography": "<comment>",
            "Themes and Messages": "<comment>",
            "Personal Impact and Enjoyment": "<comment>"
        }
    }

    Only output the list of comments in JSON format.
    """

    eval_agent_system_message = """
    You are simulating the user. Evaluate the recommended movies based on the user's preferences and the comments provided.

    **Scoring Instructions:**

    - For each movie, start with a base score of 0.
    - Add 1 point for each positive comment in the following categories:
    - **Plot and Storyline**
    - **Characters and Acting**
    - **Visual Effects and Cinematography**
    - **Themes and Messages**
    - **Personal Impact and Enjoyment**
    - Do not add points for neutral comments.
    - Subtract 1 point for each negative comment (minimum total score is 0).
    - Be strict and critical. Do not give high scores unless justified by the comments.

    **Definition of Comments:**

    - **Positive Comment**: Expresses satisfaction, enjoyment, or appreciation.
    - **Neutral Comment**: Neither positive nor negative; shows indifference.
    - **Negative Comment**: Expresses disappointment, criticism, or dislike.

    **Example Evaluation:**

    If a movie has a comment structure like this:
    ```json
    {
        "movie_title": "Example Movie",
        "comments": {
            "Plot and Storyline": "The storyline started strong but lost momentum halfway through. Some plot points felt underdeveloped.",
            "Characters and Acting": "The lead actor is OK, but supporting characters lacked depth.",
            "Visual Effects and Cinematography": "The cinematography was stunning, especially the scenes shot at sunset.",
            "Themes and Messages": "The movie touched on important themes of redemption, but didn't explore them fully.",
            "Personal Impact and Enjoyment": "Overall, I enjoyed parts of the movie but felt it didn't live up to its potential. I might not watch it again."
        }
    }

    The score would be:0 (base score) - 1 (Plot) + 0 (Characters) + 1 (Visual Effects) + 0 (Themes) + 0 (Personal Feelings) = 0

    **Instructions:**

    - **Output only the list of evaluations in valid JSON format.**
    - **Do not include any additional text or explanations.**
    - **Ensure the JSON is properly formatted and can be parsed by `json.loads()`.**

    Provide the evaluations in the following JSON format:

    ```json
    [
        {
            "movie_title": "<title>",
            "evaluation": <score>
        },
        ...
    ]
    """

    judge_agent_system_message = """
    You are the judge agent. Based on the evaluations provided, remove movies that are rated not 5. Provide a list of movies to be removed. If all movies are rated 5, indicate that the process is complete.

    Output your response in the following JSON format:

    {
        "movies_to_remove": [ "<movie_title1>", "<movie_title2>", ... ],
        "process_complete": true/false
    }

    Only output the JSON object.
    """


    analysis_agent = ConversableAgent("analysis_agent",
                                      system_message=analysis_agent_system_message,
                                      llm_config=llm_config)

    rec_agent = ConversableAgent("recommendation_agent",
                                 system_message=rec_agent_system_message,
                                 llm_config=llm_config)

    comment_simulator_agent = ConversableAgent("comment_simulator_agent",
                                               system_message=comment_simulator_system_message,
                                               llm_config=llm_config)

    eval_agent = ConversableAgent("evaluation_agent",
                                  system_message=eval_agent_system_message,
                                  llm_config=llm_config)
    judge_agent = ConversableAgent("judge_agent",
                                   system_message=judge_agent_system_message,
                                   llm_config=llm_config)


    datafetch_chat_result = entrypoint_agent.initiate_chat(fetchdata_agent, message=user_query, max_turns=2)
    user_movie_info = datafetch_chat_result.chat_history[2]['content']


    result = entrypoint_agent.initiate_chats([
        {
            "recipient": analysis_agent,
            "message": user_movie_info,
            "max_turns": 1,
            "summary_method": "last_msg",
        }
    ])
    analysis_output = result[-1].chat_history[1]['content']
    analysis_result = analysis_output

 
    movies_to_remove = []
    final_movie_list = []
    iteration = 0
    max_iterations = 3
    recommendation_history = []

    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}")


        if not movies_to_remove:
            rec_message = f"Based on the analysis results: {analysis_result}\nRecommend the top 20 movies that best align with the user's preferences. Only provide the movie names in a Python list format."
        else:
            movies_to_remove_str = json.dumps(movies_to_remove)
            rec_message = f"Based on the analysis results: {analysis_result}\nRemove the following movies from the recommendation list and replace them with new recommendations to maintain a list of 20 movies.\nMovies to remove: {movies_to_remove_str}\nOnly provide the movie names in a Python list format."
        result = entrypoint_agent.initiate_chats([
            {
                "recipient": rec_agent,
                "message": rec_message,
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ])
        rec_output = result[-1].chat_history[1]['content']


        match = re.search(r'\[.*\]', rec_output, re.DOTALL)
        if match:
            list_content = match.group(0)
            try:
                recommended_movies = json.loads(list_content)
            except json.JSONDecodeError as e:
                print("Error parsing recommended movies:", e)
                break
        else:
            print("No movie list found in rec_agent output")
            break


        if not recommended_movies:
            print("No more movies can be recommended.")
            break


        comment_message = f"Based on the user's analysis results:\n{analysis_result}\n\n.Suppose you are such a user and here are some movies you've watched:\n{json.dumps(recommended_movies)}\n\nGenerate honest and critical comments for each movie as per the system message."
        result = entrypoint_agent.initiate_chats([
            {
                "recipient": comment_simulator_agent,
                "message": comment_message,
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ])
        comments_output = result[-1].chat_history[1]['content']

        try:
            comments_output_json = re.search(r'\[.*\]', comments_output, re.DOTALL)
            if comments_output_json:
                comments_output_clean = comments_output_json.group(0)
                comments_data = json.loads(comments_output_clean)
            else:
                raise ValueError("No JSON array found in comments_output")
        except Exception as e:
            print("Error parsing comments:", e)
            break

        eval_message = f"Here are the comments for the recommended movies:\n{comments_output_clean}\n\nEvaluate these movies from the user's perspective based on the comments provided, and follow the scoring instructions in your system message."
        result = entrypoint_agent.initiate_chats([
            {
                "recipient": eval_agent,
                "message": eval_message,
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ])
        eval_output = result[-1].chat_history[1]['content']

        try:
            eval_output_json = re.search(r'\[.*\]', eval_output, re.DOTALL)
            if eval_output_json:
                eval_output_clean = eval_output_json.group(0)
                evaluations = json.loads(eval_output_clean)
            else:
                raise ValueError("No JSON array found in eval_output")
        except Exception as e:
            print("Error parsing evaluations:", e)
            break

        total_score = sum([eval['evaluation'] for eval in evaluations])
        average_score = total_score / len(evaluations)
        print(f"Average score for iteration {iteration}: {average_score}")

        recommendation_history.append({
            'iteration': iteration,
            'recommended_movies': recommended_movies,
            'average_score': average_score
        })


        judge_message = f"Here are the evaluations:\n{eval_output_clean}\n\nAs per your instructions, remove movies that are rated not 5."
        result = entrypoint_agent.initiate_chats([
            {
                "recipient": judge_agent,
                "message": judge_message,
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ])
        judge_output = result[-1].chat_history[1]['content']

        try:
            judge_output_json = json.loads(judge_output)
            movies_to_remove = judge_output_json.get('movies_to_remove', [])
            # Update the final movie list
            final_movie_list = [eval['movie_title'] for eval in evaluations]
        except Exception as e:
            print("Error parsing judge agent output:", e)
            break

 
    if recommendation_history:
        best_recommendation = max(recommendation_history, key=lambda x: x['average_score'])
        print(f"Best recommendation is from iteration {best_recommendation['iteration']} with average score {best_recommendation['average_score']}")
        final_movie_list = best_recommendation['recommended_movies']
    else:
        print("No recommendations were generated.")

    print("Final Movie List:")
    print(final_movie_list)
    return final_movie_list

# Do not modify this code below.
if __name__ == "__main__":
    main(sys.argv[1])