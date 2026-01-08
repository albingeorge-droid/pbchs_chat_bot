from __future__ import annotations

from graph import PropertyChatbotGraph


def main():
    # CLI example – treat entire terminal session as one user/thread
    chatbot = PropertyChatbotGraph(user_id="cli_user3")


    print("Punjabhi Bagh Property Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_query = input("You: ").strip()
        except EOFError:
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        # We still receive sql_query, rows if you want to log them later,
        # but we don't print them to the console.
        answer, sql_query, rows = chatbot.run(user_query)

        print("\nBot:")
        print(answer)
        print()  # just a blank line for spacing


if __name__ == "__main__":
    main()
