import pandas as pd
import psycopg2
from psycopg2 import Error as Psycopg2Error
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# PostgreSQL Odoo Connection Parameters (S-IntelliChain ERP Container)
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "sintellichain_db",
    "user": "odoo",
    "password": "odoo_password"
}

DATABASE_URL = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"


def test_method_a() -> None:
    """
    METHOD A: Pandas + SQLAlchemy
    Reads data directly into a Pandas DataFrame (structured table format).
    Highly recommended for Machine Learning or AI data pipelines.
    """
    print("\n--- [Method A] Testing Connection via Pandas & SQLAlchemy ---")
    try:
        engine = create_engine(DATABASE_URL)

        # Query retrieving POS transaction history
        query = """
        SELECT
            pol.id AS line_id,
            po.name AS order_ref,
            po.date_order AS transaction_date,
            pt.name AS product_name,
            pol.qty AS quantity_sold,
            pol.price_unit AS unit_price
        FROM pos_order_line pol
        JOIN pos_order po ON pol.order_id = po.id
        JOIN product_product pp ON pol.product_id = pp.id
        JOIN product_template pt ON pp.product_tmpl_id = pt.id
        ORDER BY po.date_order DESC;
        """

        df = pd.read_sql_query(query, engine)
        print("✓ Method A Connection Successful!")
        print(f"Total transaction rows retrieved: {len(df)}")
        if not df.empty:
            print("\nData Preview (Last 5 Transactions):")
            print(df.head())
        else:
            print("Status: Database connected, but POS transaction table is currently empty.")

    except SQLAlchemyError as err:
        print(f"✘ Method A SQLAlchemy Error: {err}")


def test_method_b() -> None:
    """
    METHOD B: psycopg2 (Native Driver)
    Uses a low-level cursor connection.
    Suitable for manual SQL execution (INSERT, UPDATE, DELETE).
    """
    print("\n--- [Method B] Testing Connection via Psycopg2 Native ---")
    try:
        conn = psycopg2.connect(
            host=str(DB_CONFIG["host"]),
            port=int(DB_CONFIG["port"]),
            dbname=str(DB_CONFIG["database"]),
            user=str(DB_CONFIG["user"]),
            password=str(DB_CONFIG["password"])
        )
        cursor = conn.cursor()

        # Query retrieving Odoo master product data
        cursor.execute("SELECT id, name FROM product_template LIMIT 5;")
        products = cursor.fetchall()

        print("✓ Method B Connection Successful!")
        print("Odoo Master Product List (Max 5):")
        for prod in products:
            print(f"- ID: {prod[0]} | Product Name: {prod[1]}")

        cursor.close()
        conn.close()

    except Psycopg2Error as err:
        print(f"✘ Method B Psycopg2 Error: {err}")


if __name__ == "__main__":
    print("==================================================")
    print(" S-INTELLICHAIN ERP - ODOO POSTGRESQL CONNECT TEST")
    print("==================================================")
    test_method_a()
    test_method_b()
