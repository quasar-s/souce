document.querySelector("button").addEventListener("click",ask)

async function ask() {
    // 사용자 질문입력=> 서버
    const question = document.querySelector("#question").value

    const response = await fetch("/api/question",{
        method:"POST",
        headers:{
            "content-type":"application/json"
        },
        body:JSON.stringify({question:question})
    })
    // 전송후 answer 도착시 answer 화면띄우기
    const answer = await response.json()

    document.querySelector("#answer").textContent = answer.message
}

// 파일 업로드
document.querySelector("#uploadBtn").addEventListener("click",uploadFile)
async function uploadFile() {
    const fileInput = document.querySelector("#file")

    const file = fileInput.files[0]

    if(!file){
        alert('파일을 선택해주세요');
        return;
    }
    const formData = new FormData();
    formData.append("file",file);


    const response = await fetch("/api/rag/upload",{
        method:"POST",
        body:formData
    })
    // 전송후 answer 도착시 answer 화면띄우기
    const answer = await response.json()
    document.querySelector('#result').textContent = answer.message
}

document.querySelector("#askBtn").addEventListener("click",rag_ask)
async function rag_ask() {
    // 사용자 질문입력=> 서버
    const question = document.querySelector("#question").value

    const response = await fetch("/api/rag/question/stream",{
        method:"POST",
        headers:{
            "content-type":"application/json"
        },
        body:JSON.stringify({question:question})
    })
    // // 전송후 answer 도착시 answer 화면띄우기
    // const answer = await response.json()

    // document.querySelector("#answer_result").textContent = answer.message

    // stream 방식
    const reader =response.body.getReader();
    const decoder = new TextDecoder();
    let answer = ""

    while(true){
        const {value, done} = await reader.read();

        if(done){
            break;
        }
        answer += decoder.decode(value);
        document.querySelector("#answer_result").textContent = answer
    }
}