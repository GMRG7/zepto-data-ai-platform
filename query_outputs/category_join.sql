SELECT c.category_name, b.title, b.rating FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY c.category_name,b.rating DESC;
