from openai import OpenAI

def gpt_prediction(preconversation, newprompt):
	new_message = {"role": "user", "content": newprompt}
	messages = preconversation.copy()
	messages.append(new_message)
 	response = client.chat.completions.create(
					messages=messages,
					model="gpt-3.5-turbo-1106",
					temperature=0.3)
	return response.choices[0].message.content

apikey = "" # you api key
client = OpenAI(api_key=apikey)

# prompt engineering : system - explain behavior
# 1-shot learning - example of input and output ander user and assistant

preconversation = [{"role":"system", "content":"You are a crystallography expert with deep knowledge on crystal structure"},
                        {"role":"user","content":"Li₂YCPO₇ crystallizes in the monoclinic P2₁/m space group."},
                        {"role":"assistant","content":"poscar"}]

new_question = "Li₂YCPO₇ crystallizes in the monoclinic P2₁/m space group."
print(gpt_prediction(preconversation, new_question))