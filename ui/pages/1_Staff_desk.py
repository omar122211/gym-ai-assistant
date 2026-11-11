"""Staff desk: report, tickets and bookings for gym staff (Streamlit multipage — appears in the sidebar).

Talks only to the FastAPI /staff endpoints, with the STAFF_TOKEN typed in the sidebar.
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
STATUS_LABELS = {"open": "🟠 Open", "in_progress": "🔵 In progress", "resolved": "🟢 Resolved"}

st.set_page_config(page_title="Fitness Abbas · Staff desk", page_icon="🛠️", layout="wide")


def staff_api(method: str, path: str, **kwargs) -> dict:
    headers = {"X-Staff-Token": st.session_state.get("staff_token", "")}
    try:
        r = httpx.request(method, f"{API_URL}{path}", headers=headers, timeout=30, **kwargs)
    except httpx.HTTPError:
        return {"error": "The API is not reachable. Is it running?"}
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        return {"error": detail if isinstance(detail, str) else f"Request failed ({r.status_code})."}
    return r.json()


with st.sidebar:
    st.header("🛠️ Staff desk")
    st.text_input("Staff token", type="password", key="staff_token", help="The STAFF_TOKEN value from the server's .env")

st.title("Staff desk")
if not st.session_state.get("staff_token"):
    st.info("Enter the staff token in the sidebar to see bookings, tickets and the report.")
    st.stop()

report = staff_api("GET", "/staff/report")
if "error" in report:
    st.error(report["error"])
    st.stop()

# ---------------------------------------------------------------- report
t, tickets_by = report["totals"], report["tickets_by_status"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Bookings", t["bookings"])
c2.metric("Open tickets", tickets_by["open"])
c3.metric("In progress", tickets_by["in_progress"])
c4.metric("Resolved", tickets_by["resolved"])

left, right = st.columns([2, 1])
with left:
    st.subheader("Bookings per class")
    per_class = report["per_class"]
    st.bar_chart({"Class": [r["class_name"] for r in per_class], "Bookings": [r["bookings"] for r in per_class]},
                 x="Class", y="Bookings", height=260)
with right:
    st.subheader("Busiest days")
    if report["busiest_days"]:
        st.dataframe([{"Day": d["day"], "Bookings": d["bookings"]} for d in report["busiest_days"]], hide_index=True)
    else:
        st.caption("No bookings yet.")

# ---------------------------------------------------------------- tickets (read + update)
st.divider()
st.subheader("Tickets")
choice = st.radio("Show", ["all", *STATUS_LABELS], horizontal=True,
                  format_func=lambda s: "All" if s == "all" else STATUS_LABELS[s])
data = staff_api("GET", "/staff/tickets", params={} if choice == "all" else {"status": choice})
if "error" in data:
    st.error(data["error"])
elif not data["tickets"]:
    st.caption("No tickets here.")
else:
    for tk in data["tickets"]:
        with st.container(border=True):
            a, b, c = st.columns([5, 2, 1])
            a.markdown(f"**#{tk['id']}** · {tk['created_at'][:16].replace('T', ' ')} · {tk['language'] or '—'}")
            a.write(tk["summary"])
            new = b.selectbox("Status", list(STATUS_LABELS), index=list(STATUS_LABELS).index(tk["status"]),
                              format_func=STATUS_LABELS.get, key=f"st{tk['id']}", label_visibility="collapsed")
            if c.button("Save", key=f"save{tk['id']}", disabled=new == tk["status"]):
                res = staff_api("PATCH", f"/staff/tickets/{tk['id']}", json={"status": new})
                st.toast(res.get("error") or f"Ticket #{tk['id']} → {STATUS_LABELS[new]}")
                st.rerun()

# ---------------------------------------------------------------- bookings (read + delete)
st.divider()
st.subheader("Bookings")
data = staff_api("GET", "/staff/bookings")
if "error" in data:
    st.error(data["error"])
elif not data["bookings"]:
    st.caption("No bookings yet.")
else:
    rows = data["bookings"]
    st.dataframe([{"Id": b["id"], "Member": b["member_name"], "Class": b["class_name"], "Day": b["day"], "Time": b["time"],
                   "Booked": b["created_at"][:16].replace("T", " ")} for b in rows], hide_index=True, width="stretch")
    with st.form("cancel"):
        bid = st.selectbox("Cancel a booking", [b["id"] for b in rows],
                           format_func=lambda i: next(f"#{b['id']} · {b['member_name']} · {b['class_name']} {b['day']} {b['time']}"
                                                      for b in rows if b["id"] == i))
        sure = st.checkbox("Yes, cancel this booking")
        if st.form_submit_button("Cancel booking", disabled=False) and sure:
            res = staff_api("DELETE", f"/staff/bookings/{bid}")
            st.toast(res.get("error") or f"Booking #{bid} cancelled")
            st.rerun()
