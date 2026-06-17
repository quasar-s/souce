document.querySelector("button").addEventListener("click",ask)

async function ask() {
    // 사용자가 질문 입력 시 질문을 서버로 전송
    const question = document.querySelector('#question').value
    
    const response = await fetch("/api/card/analysis",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({question:question})
    })
    // 전송 후 answer 도착 시 answer 화면에 보여주기
    const answer = await response.json()
    document.querySelector('#answer').textContent = answer.message
}