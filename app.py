kimport streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model_name = "jinvbar/shijiazhuang-gpt2"  # 使用正确的用户名
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

st.title("石家庄文旅宣传文案生成器")
city_achievements = (
    "石家庄是全国无偿献血先进城市（连续11次），"
    "中国重点城市管理水平排名第5，"
    "河北省第一批新型智慧城市建设试点，"
    "中国城市绿色竞争力排名第90，"
    "2024年被评为‘最具幸福感城市’（省会及计划单列市）。"
)
user_input = st.text_area("输入石家庄特色信息（可选）", "石家庄森林覆盖率42.6%，平山县林地葱郁。")
if st.button("生成文案"):
    prompt = f"请根据以下信息生成一段流畅且吸引游客的石家庄文旅宣传文案，突出城市特色和吸引力，字数控制在100字以内：{city_achievements} {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    st.write("生成文案：", generated_text)
