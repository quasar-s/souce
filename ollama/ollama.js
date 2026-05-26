async function logJSONData() {
  const response = await fetch("http://localhost:11434/api/chat", {
    method:"POST",
    headers:{"Content-Type": "application/json"},
    body:JSON.stringify({
        model:'qwen2.5:7b',
        messages:[
            {'role' : 'system', 'content' :"당신은 친절한 AI 어시스턴트입니다."},
            {'role' : 'user', 'content' : "파이썬의 장점 3개 알려줘"}
    ],
    stream:false
    })
  });
  const jsonData = await response.json();
  console.log(jsonData.message.content);
}

logJSONData()