import pickle, re

model = pickle.load(open('model/model.pkl','rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl','rb'))
print('Intercept:', model.intercept_)
print('Model type:', type(model).__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Test with realistic news-length text (not just short one-liners)
tests = [
    # REAL news - longer form
    ('Federal Reserve raises interest rates by a quarter point The Federal Reserve raised its benchmark interest rate by a quarter percentage point on Wednesday, the latest move in its campaign to bring down the highest inflation in decades. The central bank said it would continue raising rates as needed to get inflation back to its target of two percent.', 'real'),
    ('Pfizer vaccine approved by FDA for emergency use The Food and Drug Administration on Friday gave emergency authorization for the Pfizer BioNTech coronavirus vaccine, making it the first vaccine to receive the green light from regulators in the United States. The agency acted after its advisers voted to recommend authorization for people sixteen and older.', 'real'),
    ('Stock market rises as inflation data comes in lower than expected Wall Street stocks rose on Wednesday after data showed consumer price inflation was lower than expected last month, raising hopes the Federal Reserve could slow its pace of interest rate increases. The S and P 500 gained one and a half percent.', 'real'),
    # FAKE news
    ('SHOCKING TRUTH REVEALED vaccines contain 5G microchips scientists say A bombshell report has exposed what globalist elites do not want you to know. Secret documents prove that all COVID vaccines contain microscopic 5G tracking chips designed to monitor your every movement. The mainstream media is suppressing this information because they are controlled by Bill Gates.', 'fake'),
    ('NASA admits moon landing was completely staged in Hollywood The truth has finally come out after 50 years. A whistleblower from inside NASA has revealed with undeniable proof that the entire Apollo moon landing was filmed in a secret Hollywood studio by Stanley Kubrick. The government has been hiding this massive conspiracy from the American people.', 'fake'),
    ('BREAKING Aliens living among us government cover up exposed Exclusive shocking footage has leaked online showing reptilian alien humanoids walking freely in Washington DC. Multiple eyewitness accounts confirm that world leaders are secretly aliens in disguise. The deep state is desperately trying to scrub this information from the internet.', 'fake'),
]

correct = 0
print('\n--- Prediction Results (with realistic text length) ---')
for text, expected in tests:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = 'REAL' if pred == 1 else 'FAKE'
    conf = round(float(max(proba)) * 100, 2)
    ok = (pred == 1) == (expected == 'real')
    if ok:
        correct += 1
    status = '[OK]  ' if ok else '[FAIL]'
    print(status + ' [' + label + ' ' + str(conf) + '%] exp=' + expected.upper())
    print('      ' + text[:80] + '...')
    print()

print('Score: ' + str(correct) + '/' + str(len(tests)) + ' (' + str(round(correct/len(tests)*100)) + '%)')
