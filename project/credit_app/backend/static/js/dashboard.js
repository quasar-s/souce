const drawCategoryChart = (data) => {
    new Chart(document.querySelector("#categoryChart"), {
        type: 'pie',
        data: {
            labels:data.map((x) => x.category),
            datasets:[
                {data:data.map((x)=>x.amount)}
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: '카테고리별 소비내역'
                }
            }
        },
    })
}

const drawMonthlyChart = (data) => {
    new Chart(document.querySelector("#monthlyChart"), {
        type: 'line',
        data: {
            labels:data.map((x) => x.month),
            datasets:[
                {
                    label:"월별 소비",
                    data:data.map((x)=>x.amount)
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {           
                title: {
                    display: true,
                    text: '월별 소비내역'
                }
            }
        },
    })
}



async function loadDashboard() {
    const response = await fetch("/api/card/dashboard");
    const data = await response.json()
    console.log(data)

    drawCategoryChart(data.category)
    drawMonthlyChart(data.monthly)
}

window.onload = loadDashboard;