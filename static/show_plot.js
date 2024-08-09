// plot.js

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('plotForm');
    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission

        const formData = new FormData(form);
        const xhr = new XMLHttpRequest();
        xhr.open('POST', form.action, true);
        xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
        xhr.setRequestHeader('X-CSRFToken', formData.get('csrfmiddlewaretoken'));

        xhr.onload = function() {
            if (xhr.status >= 200 && xhr.status < 300) {
                const response = JSON.parse(xhr.responseText);
                const plotContainer = document.getElementById('plotContainer');
                
                // Append new plot to the container
                const newPlot = document.createElement('img');
                newPlot.src = `data:image/png;base64,${response.image_data}`;
                newPlot.alt = 'Plot';
                newPlot.className = 'plot-img';

                plotContainer.appendChild(newPlot);
            } else {
                console.error('Request failed with status:', xhr.status);
            }
        };

        xhr.send(formData);
    });
});
