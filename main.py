from __future__ import annotations

from graph import PropertyChatbotGraph


def main():
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

        # Buffer to know whether anything was streamed
        streamed_tokens: list[str] = []
        first_token = True

        def print_token(token: str) -> None:
            """Streaming callback used only when build_answer calls the LLM."""
            nonlocal first_token
            if first_token:
                # Print Bot: only once, on the first token
                print("\nBot: ", end="", flush=True)
                first_token = False
            print(token, end="", flush=True)
            streamed_tokens.append(token)

        # Run the graph (may or may not use streaming depending on the node)
        answer, sql_query, rows, geometry = chatbot.run(
            user_query,
            on_token=print_token,
        )

        # If NOTHING was streamed (note-summary, map, small-talk, errors, etc.),
        # fall back to printing the final_answer string once.
        if not streamed_tokens:
            print("\nBot: ", end="", flush=True)
            if answer:
                print(answer, end="", flush=True)


        # For debugging / full-stack:
        #   - rows will contain pbchs_map rows when map is available
        #   - geometry is a list of GeoJSON geometry objects
        if geometry:
            print(f"\n[DEBUG] Geometry features returned: {len(geometry)}", end="")

        # Finish the line after any streaming or fallback
        print("\n")

if __name__ == "__main__":
    main()
