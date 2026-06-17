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


    const response = await fetch("/api/card/upload",{
        method:"POST",
        body:formData
    })
    // 전송후 answer 도착시 answer 화면띄우기
    const answer = await response.json()
    document.querySelector('#result').textContent = answer.message
}