const canvas = document.getElementById("canvas");

const ctx = canvas.getContext("2d");

ctx.fillStyle = "white";
ctx.fillRect(0,0,280,280);

ctx.strokeStyle="black";
ctx.lineWidth=18;
ctx.lineCap="round";

let drawing=false;

canvas.addEventListener("mousedown",()=>{

    drawing=true;

});

canvas.addEventListener("mouseup",()=>{

    drawing=false;

    ctx.beginPath();

});

canvas.addEventListener("mousemove",draw);

function draw(e){

    if(!drawing) return;

    const rect=canvas.getBoundingClientRect();

    const x=e.clientX-rect.left;
    const y=e.clientY-rect.top;

    ctx.lineTo(x,y);

    ctx.stroke();

    ctx.beginPath();

    ctx.moveTo(x,y);

}

document
.getElementById("clearBtn")
.addEventListener("click",()=>{

    ctx.fillStyle="white";

    ctx.fillRect(0,0,280,280);

});

// Predict Button
document.getElementById("predictBtn").addEventListener("click", async () => {

    // Convert canvas to image
    const image = canvas.toDataURL("image/png");

    // Send image to Flask
    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            image: image
        })
    });

    const result = await response.json();

    document.getElementById("prediction").innerText =
        "Prediction: " + result.prediction;

});