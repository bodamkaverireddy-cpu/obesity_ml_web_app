// Table scrolling utilities for large datasets

function makeTableScrollable(tableId) {
    const table = document.getElementById(tableId);
    if (table) {
        const wrapper = document.createElement('div');
        wrapper.style.overflowX = 'auto';
        wrapper.style.maxWidth = '100%';
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const tables = document.querySelectorAll('.data-table');
    tables.forEach(table => {
        if (!table.id) {
            table.id = 'table-' + Math.random().toString(36).substr(2, 9);
        }
        makeTableScrollable(table.id);
    });
});

