document.getElementById('symptom-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const symptoms = document.getElementById('symptoms').value;
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({ symptoms }),
    });

    const resultDiv = document.getElementById('result');
    if (response.ok) {
        const data = await response.json();
        resultDiv.textContent = `Predicted Disease: ${data.predicted_disease}`;
    } else {
        const error = await response.json();
        resultDiv.textContent = `Error: ${error.error}`;
    }
});
