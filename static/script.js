document.getElementById('loanForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    // Convert types
    data.person_age = parseFloat(data.person_age);
    data.person_income = parseFloat(data.person_income);
    data.person_emp_exp = parseInt(data.person_emp_exp);
    data.loan_amnt = parseFloat(data.loan_amnt);
    data.loan_int_rate = parseFloat(data.loan_int_rate);
    data.loan_percent_income = parseFloat(data.loan_percent_income);
    data.cb_person_cred_hist_length = parseFloat(data.cb_person_cred_hist_length);
    data.credit_score = parseInt(data.credit_score);
    data.previous_loan_defaults_on_file = formData.get('previous_loan_defaults_on_file') === 'on';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Display results
        const resultDiv = document.getElementById('result');
        const decisionBox = document.getElementById('decisionBox');
        const decisionText = document.getElementById('decisionText');
        const probabilityText = document.getElementById('probability');
        const reasonText = document.getElementById('reason');

        resultDiv.classList.remove('hidden');
        decisionText.textContent = result.decision;
        probabilityText.textContent = (result.probability * 100).toFixed(2) + '%';
        reasonText.textContent = result.reason;

        decisionBox.className = 'decision-box'; // Reset classes
        if (result.decision === 'APPROVED') {
            decisionBox.classList.add('approved');
            decisionText.textContent = 'APPROUVÉ';
        } else {
            decisionBox.classList.add('rejected');
            decisionText.textContent = 'REJETÉ';
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Une erreur est survenue lors de l\'analyse.');
    }
});
