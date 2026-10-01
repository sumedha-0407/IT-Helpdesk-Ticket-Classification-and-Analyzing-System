"""Generate a noisy synthetic IT helpdesk ticket dataset (train/val/test split by unique text)."""
import random, numpy as np, pandas as pd
random.seed(7); np.random.seed(7)
N = 6000

# category -> issue -> phrasings.  Some issues deliberately overlap across categories.
ISSUES = {
 "Network": {
  "VPN drops": ["vpn keeps disconnecting", "vpn connection times out", "cannot stay connected to the vpn"],
  "WiFi down": ["wifi is very slow", "cannot connect to office wifi", "wireless keeps dropping"],
  "Shared drive": ["cannot open the shared drive", "network folder not accessible", "mapped drive missing"],
  "No internet": ["no internet on my desk", "internet not working", "pages will not load, network issue"]},
 "Hardware": {
  "Laptop fault": ["laptop will not turn on", "laptop screen flickering", "laptop battery drains quickly"],
  "Printer": ["printer is not printing", "printer shows offline", "paper jam on the floor printer"],
  "Peripheral": ["keyboard stopped working", "mouse is not responding", "external monitor has no signal"],
  "Slow machine": ["computer is extremely slow", "laptop freezes all the time", "pc very slow and hot"]},
 "Software": {
  "App crash": ["excel keeps crashing", "application crashes on startup", "word freezes when saving"],
  "Install": ["need software installed", "cannot install the update", "installation fails with error"],
  "Licence": ["licence expired for the tool", "software asks for activation", "subscription not recognised"],
  "Slow app": ["application is very slow", "crm takes forever to load", "software hangs when opening files"]},
 "Access/Account": {
  "Password": ["forgot my password", "password reset needed", "account locked after wrong password"],
  "Login": ["cannot login to my account", "login fails with error", "unable to sign in to the portal"],
  "Permission": ["need access to the folder", "permission denied on the system", "please grant me access to the tool"],
  "MFA": ["mfa code not arriving", "authenticator app not working", "new phone need to set up mfa"]},
 "Email": {
  "Not syncing": ["email not syncing on phone", "outlook not receiving emails", "mailbox not updating"],
  "Full mailbox": ["mailbox is full", "cannot send emails storage limit", "outlook says quota exceeded"],
  "Spam": ["too much spam in inbox", "legitimate emails going to junk", "emails bouncing back"],
  "Calendar": ["calendar invites not showing", "meeting invite missing in outlook", "shared calendar not updating"]},
 "Security": {
  "Phishing": ["received a suspicious email", "got a phishing message asking for password", "strange email with link"],
  "Malware": ["antivirus found a virus", "suspicious popup keeps appearing", "computer behaving strangely possible malware"],
  "Lost device": ["lost my company laptop", "phone with work email stolen", "laptop left in taxi"],
  "Unauthorised": ["unusual login alert on my account", "someone tried to access my account", "unknown device signed in"]},
}
# ambiguous confusers: (category, issue) -> alternative category annotators may pick
CONFUSE = {("Network","Shared drive"):"Access/Account", ("Hardware","Slow machine"):"Software",
           ("Email","Not syncing"):"Software", ("Security","Malware"):"Software",
           ("Access/Account","Login"):"Email", ("Hardware","Printer"):"Network",
           ("Software","Slow app"):"Network"}
PREFIX = ["", "", "hi team,", "hello,", "urgent:", "please help,", "since this morning", "again today,", "issue:", "hey,"]
IMPACT = {"low impact, not urgent": 0, "only affects me, minor": 0, "single user affected": 1, "blocking my work": 2,
          "whole team affected": 2, "all users on my floor affected": 2, "need this fixed asap": 2,
          "production is affected": 3, "company wide outage": 3, "critical, all users down": 3}
SUFFIX = ["", "", "please advise", "thanks", "not sure what happened", "third time this week", "started yesterday",
          "tried restarting already", "whenever you can"] + list(IMPACT)
HI = ["asap","production","all users","blocking","urgent","stolen","phishing","virus","lost"]
LO = ["not urgent","low impact","whenever you can"]
DEPTS = ["Finance","HR","Sales","Engineering","Marketing","Operations","Legal","Support"]
DEPT_W = [0.12,0.06,0.2,0.25,0.1,0.1,0.04,0.13]
PRIO = ["Low","Medium","High","Critical"]
BASE_P = {"Network":[.2,.4,.3,.1],"Hardware":[.3,.45,.2,.05],"Software":[.35,.45,.17,.03],
          "Access/Account":[.3,.45,.22,.03],"Email":[.4,.4,.17,.03],"Security":[.05,.2,.4,.35]}
RES_H = {"Network":8,"Hardware":30,"Software":14,"Access/Account":3,"Email":6,"Security":20}
ACTION = {"Network":["Restarted service","Reconfigured router/VPN","Escalated to network team"],
          "Hardware":["Replaced part","Swapped device","Vendor repair"],
          "Software":["Reinstalled app","Applied patch","Renewed licence"],
          "Access/Account":["Reset password","Granted permission","Reset MFA"],
          "Email":["Cleared mailbox","Re-synced profile","Adjusted spam rules"],
          "Security":["Isolated device","Blocked sender","Forced credential reset"]}
OFFVOC = ("lunch party catering parking holiday birthday cake gym coffee machine desk chair plant window travel visa budget "
          "invoice reimbursement banana keyboard random test ignore weekend football cinema ticket garden recipe pizza "
          "meeting room booking decoration balloon gift voucher yoga parking permit vending snack fridge microwave").split()
ODD_OLD = ["order catering for the office party", "asdf qwer test test 123", "where is the new coffee machine located",
       "can you book a meeting room in building c for friday lunch", "xkcd zzzz random words banana keyboard",
       "question about the holiday policy and parking permits", "ticket test please ignore", "quarterly budget spreadsheet template request for travel reimbursements and visas"]

def typo(s):
    w = s.split()
    for i in range(len(w)):
        if len(w[i]) > 4 and random.random() < 0.05:
            j = random.randrange(len(w[i])-1); w[i] = w[i][:j]+w[i][j+1]+w[i][j]+w[i][j+2:]
    return " ".join(w)

rows = []
cats = list(ISSUES)
while len(rows) < N:
    if random.random() < 0.025:
        cat = random.choice(cats); issue = "Miscellaneous"; text = " ".join(random.sample(OFFVOC, random.randint(4, 8))); odd = 1
        if random.random() < .5: text = f"{random.choice(PREFIX)} {text} {random.choice(SUFFIX)}".strip()
    else:
        cat = random.choice(cats); issue = random.choice(list(ISSUES[cat]))
        text = f"{random.choice(PREFIX)} {random.choice(ISSUES[cat][issue])} {random.choice(SUFFIX)}".strip(); odd = 0
    text = typo(" ".join(text.split()))
    label = cat
    if (cat, issue) in CONFUSE and random.random() < 0.25: label = CONFUSE[(cat, issue)]
    if random.random() < 0.05: label = random.choice(cats)          # annotation noise
    p = np.array(BASE_P[cat], float)
    p /= p.sum(); pr = np.random.choice(4, p=p)
    for k, lv in IMPACT.items():
        if k in text and random.random() < 0.9: pr = lv
    if random.random() < 0.05: pr = int(np.clip(pr + random.choice([-1,1]), 0, 3))  # priority noise
    hours = round(float(np.random.lognormal(np.log(RES_H[cat]), 0.6) * [0.6,1,1.4,0.8][pr]), 1)
    rows.append(dict(ticket_id=0, created_at=pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=int(np.random.randint(0, 60*24*365))),
                     department=np.random.choice(DEPTS, p=DEPT_W), description=text, issue=issue,
                     category=label, priority=PRIO[pr], resolution_hours=hours,
                     resolution_action=random.choice(ACTION[cat]), true_outlier=odd))
df = pd.DataFrame(rows).drop_duplicates("description").sample(frac=1, random_state=1).reset_index(drop=True)
df["ticket_id"] = [f"INC{100000+i}" for i in range(len(df))]
n = len(df); df["split"] = ["train"]*int(n*.7) + ["val"]*int(n*.15) + ["test"]*(n-int(n*.7)-int(n*.15))
df.to_csv("data/tickets.csv", index=False)
print(n, "unique tickets"); print(df.split.value_counts().to_dict())
