

client = OpenAI()

# 天气工具
def call_weather_tool(city):
    from weather_backend import get_weather
    return get_weather(city)

# 工具循环调用
def chat_with_tools(user_question):

    messages = [
        {
            "role": "system",
            "content": "你是一个智能助手，可以调用天气查询工具解决用户问题。"
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    # 循环调用工具
    while True:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查询城市天气",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "城市名称"
                                }
                            },
                            "required": ["city"]
                        }
                    }
                }
            ]
        )


        message = response.choices[0].message


        # 没有工具调用，直接返回答案
        if not message.tool_calls:

            return message.content



        # 保存模型请求
        messages.append(message)



        # 执行工具
        for tool_call in message.tool_calls:

            function_name = tool_call.function.name

            args = json.loads(
                tool_call.function.arguments
            )


            if function_name == "get_weather":

                result = call_weather_tool(
                    args["city"]
                )


            # 把工具结果加入上下文
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
            )
