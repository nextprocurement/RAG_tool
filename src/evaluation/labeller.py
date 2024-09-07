
def load_prompt_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()
    return file_contents

tpc_labels = []
for tpc in topic_keys:
    this_tpc_promt = prompt_template.format(tpc)
    print(f"Topic: {tpc}")
    llm_response = gpt_model.prompt_gpt(
        prompt=this_tpc_promt, model_engine='gpt-3.5-turbo', temperature=0, max_tokens=500
    )
    time.sleep(1)
    tpc_labels.append(llm_response)
    print(f"Label: {llm_response}")

df.loc[:, "label"] = df["main_topic"].apply(lambda x: tpc_labels[x])