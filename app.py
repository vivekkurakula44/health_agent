from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import answer_query


app = FastAPI(title="Arogyaverse Health AI")


class Question(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Arogyaverse Health AI</title>

        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f7fa;
            }

            h1 {
                text-align: center;
                color: #222;
            }

            p {
                text-align: center;
                color: #555;
            }

            textarea {
                width: 100%;
                height: 120px;
                padding: 12px;
                font-size: 16px;
                box-sizing: border-box;
                border: 1px solid #ccc;
                border-radius: 8px;
                resize: vertical;
            }

            button {
                display: block;
                margin: 15px auto;
                padding: 12px 30px;
                font-size: 16px;
                cursor: pointer;
                border: none;
                border-radius: 8px;
                background: #2563eb;
                color: white;
            }

            button:hover {
                background: #1d4ed8;
            }

            #answer {
                margin-top: 25px;
                padding: 20px;
                background: white;
                border-radius: 8px;
                white-space: pre-wrap;
                min-height: 50px;
                border: 1px solid #ddd;
            }
        </style>
    </head>

    <body>

        <h1>🩺 Arogyaverse Health AI</h1>

        <p>
            Ask a health-related question in English, Hindi, or Telugu.
        </p>

        <textarea
            id="question"
            placeholder="Example: What are the symptoms of diabetes?"
        ></textarea>

        <button onclick="askQuestion()">
            Ask AI
        </button>

        <div id="answer">
            Your answer will appear here...
        </div>


        <script>

            async function askQuestion() {

                const question =
                    document.getElementById("question").value;

                const answerBox =
                    document.getElementById("answer");


                if (!question.trim()) {

                    answerBox.textContent =
                        "Please enter a question.";

                    return;
                }


                answerBox.textContent =
                    "Thinking...";


                try {

                    const response = await fetch("/ask", {

                        method: "POST",

                        headers: {
                            "Content-Type": "application/json"
                        },

                        body: JSON.stringify({
                            question: question
                        })

                    });


                    const data = await response.json();


                    if (!response.ok) {

                        answerBox.textContent =
                            "Error: " +
                            (data.detail || "Something went wrong.");

                        return;
                    }


                    answerBox.textContent =
                        data.answer || "No answer returned.";

                }

                catch (error) {

                    answerBox.textContent =
                        "Something went wrong: " + error;

                }

            }

        </script>

    </body>
    </html>
    """


@app.post("/ask")
def ask(question: Question):

    result = answer_query(question.question)

    return {
        "answer": result.get(
            "final_answer",
            "No answer generated."
        )
    }