SELECT c.category_name, COUNT(*) AS book_count FROM books b JOIN categories c ON b.category_id=c.category_id GROUP BY c.category_id,c.category_name ORDER BY book_count DESC;
