import tkinter as tk
from tkinter import ttk, messagebox
import mysql.connector
from datetime import datetime

# ---------------- DATABASE CONNECTION ----------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Saurav@123",     # <-- your MySQL root password (change if needed)
        database="rms_db_2025"
    )


# ---------------- MAIN APP CLASS ----------------
class RestaurantApp:
    def __init__(self, user_id, username):
        self.user_id = user_id
        self.username = username
        self.root = tk.Tk()
        self.root.title("Restaurant Management System")
        self.root.geometry("820x560")
        self.root.resizable(False, False)
        self.dashboard()
        self.root.mainloop()

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # ---------- DASHBOARD ----------
    def dashboard(self):
        self.clear()

        header = tk.Frame(self.root, bg="#2E8B57", height=70)
        header.pack(fill="x")
        tk.Label(header, text=f"Welcome, {self.username}", bg="#2E8B57", fg="white",
                 font=("Helvetica", 16, "bold")).pack(pady=18, anchor="w", padx=20)

        body = tk.Frame(self.root, bg="#F7F7F7")
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Left menu buttons
        left_frame = tk.Frame(body, bg="#F7F7F7", width=220)
        left_frame.pack(side="left", fill="y", padx=(8,12))
        tk.Button(left_frame, text="Place Order", bg="#2E8B57", fg="white",
                  font=("Arial", 12, "bold"), width=22, height=2, command=self.place_order).pack(pady=8)
        tk.Button(left_frame, text="View Order History", bg="#2E8B57", fg="white",
                  font=("Arial", 12, "bold"), width=22, height=2, command=self.history_screen).pack(pady=8)
        tk.Button(left_frame, text="Reservations", bg="#2E8B57", fg="white",
                  font=("Arial", 12, "bold"), width=22, height=2, command=self.reservation_screen).pack(pady=8)
        tk.Button(left_frame, text="Manage Tables", bg="#2E8B57", fg="white",
                  font=("Arial", 12, "bold"), width=22, height=2, command=self.manage_tables).pack(pady=8)
        tk.Button(left_frame, text="Logout", bg="#FF6347", fg="white",
                  font=("Arial", 12, "bold"), width=22, height=2, command=self.logout).pack(pady=8)

        # Right: Quick stats and latest orders
        right_frame = tk.Frame(body, bg="#FFFFFF")
        right_frame.pack(side="left", fill="both", expand=True, padx=(12,8))

        tk.Label(right_frame, text="Quick Stats", bg="white", font=("Helvetica", 13, "bold")).pack(anchor="nw", pady=(6,6), padx=8)
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM rms_orders")
            total_orders = cur.fetchone()[0]
            cur.execute("SELECT IFNULL(SUM(total_amount),0) FROM rms_orders")
            total_revenue = cur.fetchone()[0]
            conn.close()
        except Exception:
            total_orders = "N/A"
            total_revenue = "N/A"

        stats_frame = tk.Frame(right_frame, bg="white")
        stats_frame.pack(pady=8, anchor="nw", padx=8)
        tk.Label(stats_frame, text=f"Total Orders: {total_orders}", bg="white", font=("Arial", 11)).pack(anchor="w", pady=3)
        tk.Label(stats_frame, text=f"Total Revenue: ₹{total_revenue}", bg="white", font=("Arial", 11)).pack(anchor="w", pady=3)

        # Recent orders list
        tk.Label(right_frame, text="Your Recent Orders", bg="white", font=("Helvetica", 12, "bold")).pack(anchor="nw", pady=(12,6), padx=8)
        tree = ttk.Treeview(right_frame, columns=("Order ID", "Amount", "Date"), show="headings", height=8)
        tree.heading("Order ID", text="Order ID")
        tree.heading("Amount", text="Amount (₹)")
        tree.heading("Date", text="Date")
        tree.pack(padx=8, pady=(0,8), fill="x")
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT order_id, total_amount, order_date FROM rms_orders WHERE user_id=%s ORDER BY order_date DESC LIMIT 5", (self.user_id,))
            for r in cur.fetchall():
                tree.insert("", "end", values=r)
            conn.close()
        except Exception:
            pass

    def logout(self):
        answer = messagebox.askyesno("Confirm Logout", "Are you sure you want to logout?")
        if answer:
            self.root.destroy()
            login_window()

    # ---------- PLACE ORDER ----------
    def place_order(self):
        self.clear()
        header = tk.Frame(self.root, bg="#2E8B57", height=60)
        header.pack(fill="x")
        tk.Label(header, text="Place Order", bg="#2E8B57", fg="white",
                 font=("Helvetica", 14, "bold")).pack(pady=12)

        body = tk.Frame(self.root, bg="#FFFFFF")
        body.pack(fill="both", expand=True, padx=12, pady=10)

        # fetch menu and tables
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT menu_id, name, price FROM rms_menu_items")
            items = cur.fetchall()
            cur.execute("SELECT table_id, table_name, seats FROM rms_tables")
            tables = cur.fetchall()
            conn.close()
        except Exception as e:
            messagebox.showerror("DB Error", f"Could not fetch menu/tables:\n{e}")
            self.dashboard()
            return

        # choose table
        top_frame = tk.Frame(body, bg="white")
        top_frame.pack(fill="x", pady=(6,8))
        tk.Label(top_frame, text="Select Table (optional):", bg="white").pack(side="left", padx=6)
        table_var = tk.StringVar()
        table_combo = ttk.Combobox(top_frame, textvariable=table_var, state="readonly", width=20)
        table_combo['values'] = ["None"] + [f"{t[0]} - {t[1]} ({t[2]} seats)" for t in tables]
        table_combo.current(0)
        table_combo.pack(side="left", padx=6)

        # items list with qty entries
        canvas = tk.Canvas(body, bg="white")
        scrollbar = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="white")
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set, height=320)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        quantities = {}
        for i, (menu_id, name, price) in enumerate(items):
            tk.Label(scroll_frame, text=f"{name} - ₹{price}", anchor="w", bg="white").grid(row=i, column=0, sticky="w", padx=10, pady=6)
            qty = tk.Entry(scroll_frame, width=6)
            qty.grid(row=i, column=1, padx=8, pady=6)
            quantities[menu_id] = (qty, price)

        def submit_order():
            try:
                conn = get_connection()
                cur = conn.cursor()
                total = 0
                order_details = []
                for menu_id, (qty_entry, price_each) in quantities.items():
                    try:
                        q = int(qty_entry.get())
                    except Exception:
                        q = 0
                    if q > 0:
                        total += price_each * q
                        order_details.append((menu_id, q, price_each))

                if total == 0:
                    messagebox.showwarning("Warning", "No items selected.")
                    conn.close()
                    return

                selected = table_var.get()
                table_id = None
                if selected and selected != "None":
                    table_id = int(selected.split(" - ")[0])

                order_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cur.execute("INSERT INTO rms_orders (user_id, table_id, total_amount, order_date) VALUES (%s, %s, %s, %s)",
                            (self.user_id, table_id, total, order_date))
                order_id = cur.lastrowid

                for menu_id, qty, price_each in order_details:
                    cur.execute("INSERT INTO rms_order_lines (order_id, menu_id, quantity, price_each) VALUES (%s, %s, %s, %s)",
                                (order_id, menu_id, qty, price_each))

                conn.commit()
                conn.close()
                messagebox.showinfo("Success", f"Order placed successfully!\nTotal: ₹{total:.2f}")
                self.dashboard()
            except Exception as e:
                messagebox.showerror("Database Error", f"An error occurred:\n{e}")

        btn_frame = tk.Frame(body, bg="white")
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Submit Order", bg="#2E8B57", fg="white", font=("Arial", 10, "bold"),
                  width=18, command=submit_order).pack(side="left", padx=8)
        tk.Button(btn_frame, text="Back", bg="#FF6347", fg="white", font=("Arial", 10, "bold"),
                  width=12, command=self.dashboard).pack(side="left", padx=8)

    # ---------- ORDER HISTORY ----------
    def history_screen(self):
        self.clear()
        header = tk.Frame(self.root, bg="#2E8B57", height=60)
        header.pack(fill="x")
        tk.Label(header, text="Order History", bg="#2E8B57", fg="white",
                 font=("Helvetica", 14, "bold")).pack(pady=12)

        body = tk.Frame(self.root, bg="white")
        body.pack(fill="both", expand=True, padx=12, pady=10)

        tree = ttk.Treeview(body, columns=("Order ID", "Amount", "Date"), show="headings")
        tree.heading("Order ID", text="Order ID")
        tree.heading("Amount", text="Total Amount (₹)")
        tree.heading("Date", text="Order Date")
        tree.pack(pady=10, fill="both", expand=True)

        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT order_id, total_amount, order_date FROM rms_orders WHERE user_id=%s ORDER BY order_date DESC", (self.user_id,))
            for row in cur.fetchall():
                tree.insert("", "end", values=row)
            conn.close()
        except Exception as e:
            messagebox.showerror("Database Error", f"An error occurred:\n{e}")

        tk.Button(body, text="Back", bg="#FF6347", fg="white", font=("Arial", 10, "bold"),
                  width=12, command=self.dashboard).pack(pady=6)

    # ---------- RESERVATIONS ----------
    def reservation_screen(self):
        self.clear()
        header = tk.Frame(self.root, bg="#2E8B57", height=60)
        header.pack(fill="x")
        tk.Label(header, text="Reservations", bg="#2E8B57", fg="white",
                 font=("Helvetica", 14, "bold")).pack(pady=12)

        body = tk.Frame(self.root, bg="white")
        body.pack(fill="both", expand=True, padx=12, pady=10)

        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT table_id, table_name, seats FROM rms_tables")
            tables = cur.fetchall()
            conn.close()
        except Exception as e:
            messagebox.showerror("DB Error", f"Error getting tables:\n{e}")
            self.dashboard()
            return

        tk.Label(body, text="Choose Table:", bg="white").pack(anchor="w", padx=8)
        table_var = tk.StringVar()
        table_combo = ttk.Combobox(body, textvariable=table_var, state="readonly", width=30)
        table_combo['values'] = [f"{t[0]} - {t[1]} ({t[2]} seats)" for t in tables]
        if tables:
            table_combo.current(0)
        table_combo.pack(padx=8, pady=6)

        tk.Label(body, text="Reservation Date & Time (YYYY-MM-DD HH:MM):", bg="white").pack(anchor="w", padx=8)
        dt_entry = tk.Entry(body, width=30)
        dt_entry.pack(padx=8, pady=4)

        tk.Label(body, text="Number of Guests:", bg="white").pack(anchor="w", padx=8)
        guests_entry = tk.Entry(body, width=10)
        guests_entry.pack(padx=8, pady=4)

        tk.Label(body, text="Notes (optional):", bg="white").pack(anchor="w", padx=8)
        notes_entry = tk.Entry(body, width=50)
        notes_entry.pack(padx=8, pady=4)

        def make_reservation():
            selected = table_var.get()
            if not selected:
                messagebox.showwarning("Warning", "Select a table first.")
                return
            table_id = int(selected.split(" - ")[0])
            dt_text = dt_entry.get().strip()
            guests = guests_entry.get().strip()
            try:
                guests = int(guests)
            except Exception:
                messagebox.showwarning("Warning", "Guests must be a number.")
                return
            try:
                resv_dt = datetime.strptime(dt_text, "%Y-%m-%d %H:%M")
            except Exception:
                messagebox.showwarning("Warning", "Date/time format incorrect.")
                return
            notes = notes_entry.get().strip()
            try:
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("INSERT INTO rms_reservations (user_id, table_id, resv_datetime, guests, notes) VALUES (%s, %s, %s, %s, %s)",
                            (self.user_id, table_id, resv_dt.strftime("%Y-%m-%d %H:%M:%S"), guests, notes))
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", "Reservation created.")
                self.dashboard()
            except Exception as e:
                messagebox.showerror("DB Error", f"Could not create reservation:\n{e}")

        tk.Button(body, text="Reserve", bg="#2E8B57", fg="white", width=16, command=make_reservation).pack(pady=8)
        tk.Button(body, text="Back", bg="#FF6347", fg="white", width=12, command=self.dashboard).pack()

    # ---------- MANAGE TABLES ----------
    def manage_tables(self):
        self.clear()
        header = tk.Frame(self.root, bg="#2E8B57", height=60)
        header.pack(fill="x")
        tk.Label(header, text="Manage Tables", bg="#2E8B57", fg="white",
                 font=("Helvetica", 14, "bold")).pack(pady=12)

        body = tk.Frame(self.root, bg="white")
        body.pack(fill="both", expand=True, padx=12, pady=10)

        tree = ttk.Treeview(body, columns=("Table ID", "Name", "Seats", "Status"), show="headings")
        tree.heading("Table ID", text="Table ID")
        tree.heading("Name", text="Name")
        tree.heading("Seats", text="Seats")
        tree.heading("Status", text="Status")
        tree.pack(fill="both", expand=True, padx=8, pady=6)

        def load_tables():
            for i in tree.get_children():
                tree.delete(i)
            try:
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("SELECT table_id, table_name, seats, status FROM rms_tables")
                for r in cur.fetchall():
                    tree.insert("", "end", values=r)
                conn.close()
            except Exception as e:
                messagebox.showerror("DB Error", f"Could not load tables:\n{e}")

        def toggle_status():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("Warning", "Select a table row.")
                return
            vals = tree.item(sel[0])['values']
            tid, _, _, status = vals
            new_status = "available" if status != "available" else "occupied"
            try:
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("UPDATE rms_tables SET status=%s WHERE table_id=%s", (new_status, tid))
                conn.commit()
                conn.close()
                load_tables()
            except Exception as e:
                messagebox.showerror("DB Error", f"Could not update table status:\n{e}")

        load_tables()
        btn_frame = tk.Frame(body, bg="white")
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Toggle Status", bg="#2E8B57", fg="white", width=14, command=toggle_status).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Back", bg="#FF6347", fg="white", width=12, command=self.dashboard).pack(side="left", padx=6)

# ---------------- LOGIN / SIGNUP SCREEN ----------------
def login_window():
    root = tk.Tk()
    root.title("Restaurant Management System - Login")
    root.geometry("420x360")
    root.resizable(False, False)

    header = tk.Frame(root, bg="#2E8B57", height=60)
    header.pack(fill="x")
    tk.Label(header, text="RESTAURANT MANAGEMENT", bg="#2E8B57", fg="white", font=("Helvetica", 13, "bold")).pack(pady=14)

    body = tk.Frame(root, bg="white")
    body.pack(fill="both", expand=True, padx=10, pady=8)

    tk.Label(body, text="Username", bg="white").pack(anchor="w", pady=(6,0))
    username = tk.Entry(body, width=30)
    username.pack()

    tk.Label(body, text="Password", bg="white").pack(anchor="w", pady=(8,0))
    password = tk.Entry(body, width=30, show="*")
    password.pack()

    def login_action():
        u, p = username.get().strip(), password.get().strip()
        if not u or not p:
            messagebox.showwarning("Warning", "All fields are required.")
            return
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT user_id, username FROM rms_users WHERE username=%s AND password=%s", (u, p))
            user = cur.fetchone()
            conn.close()
        except Exception as e:
            messagebox.showerror("DB Error", f"Could not connect to database:\n{e}")
            return

        if user:
            root.destroy()
            RestaurantApp(user[0], user[1])
        else:
            messagebox.showerror("Error", "Invalid username or password!")

    def signup_action():
        u, p = username.get().strip(), password.get().strip()
        if not u or not p:
            messagebox.showwarning("Warning", "All fields are required.")
            return
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("INSERT INTO rms_users (username, password) VALUES (%s, %s)", (u, p))
            conn.commit()
            conn.close()
            messagebox.showinfo("Success", "Account created. You can now login.")
        except mysql.connector.Error as err:
            messagebox.showerror("Error", f"Could not create account:\n{err}")

    btn_frame = tk.Frame(body, bg="white")
    btn_frame.pack(pady=12)
    tk.Button(btn_frame, text="Login", bg="#2E8B57", fg="white", width=12, command=login_action).pack(side="left", padx=8)
    tk.Button(btn_frame, text="Sign Up", bg="#2E8B57", fg="white", width=12, command=signup_action).pack(side="left", padx=8)

    root.config(bg="white")
    root.mainloop()

if __name__ == "__main__":
    login_window()
