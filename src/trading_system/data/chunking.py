from typing import List

class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of maximum size `chunk_size` characters,
        with an overlap of `chunk_overlap` characters.
        Tries to break at sentence boundaries ('. ', '! ', '? ') or at least word boundaries (' ').
        """
        if not text:
            return []

        # Simple character-based chunking with boundary awareness
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # If the remaining text fits in one chunk
            if text_length - start <= self.chunk_size:
                chunks.append(text[start:])
                break

            # Find the best end position for the current chunk
            end = start + self.chunk_size

            # Try to find a sentence boundary
            best_end = -1
            for separator in ['. ', '! ', '? ', '\n']:
                sep_pos = text.rfind(separator, start + self.chunk_size // 2, end)
                if sep_pos != -1:
                    best_end = max(best_end, sep_pos + 1) # include the separator mark but not the space

            if best_end != -1:
                end = best_end
            else:
                # Fallback to word boundary
                word_bound = text.rfind(' ', start + self.chunk_size // 2, end)
                if word_bound != -1:
                    end = word_bound

            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            # We want to overlap from the end of the current chunk
            # But ensure we are moving forward to avoid infinite loop
            next_start = end - self.chunk_overlap

            # If overlap takes us back before or at the start, just advance by 1
            if next_start <= start:
                next_start = start + 1

            # If we're inside a word, try to move to the beginning of the word to avoid breaking words in overlap
            if next_start > 0 and next_start < text_length and text[next_start - 1] != ' ':
                word_start = text.rfind(' ', start, next_start)
                if word_start != -1:
                    next_start = word_start + 1

            start = next_start

        return chunks
