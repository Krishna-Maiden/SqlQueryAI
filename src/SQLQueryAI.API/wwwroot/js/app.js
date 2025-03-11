$(document).ready(function () {
    // Handle query submission
    $('#submitQuery').click(function () {
        submitQuery();
    });

    // Submit on Enter key
    $('#queryInput').keypress(function (e) {
        if (e.which === 13) {
            submitQuery();
        }
    });

    // Example query click handler
    $('.example-query').click(function () {
        $('#queryInput').val($(this).text());
        submitQuery();
    });

    function submitQuery() {
        const query = $('#queryInput').val().trim();
        if (query.length === 0) return;

        // Show loading indicator
        $('.loading').show();
        $('.result-container').addClass('d-none');

        // Call API
        $.ajax({
            url: '/api/query',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ query: query }),
            success: function (response) {
                // Hide loading indicator
                $('.loading').hide();
                $('.result-container').removeClass('d-none');

                // Format and display JSON result
                const formattedJson = JSON.stringify(response.result, null, 2);
                $('#resultDisplay').text(formattedJson);

                // Display processing time
                $('.processing-time').text(`${response.processingTimeMs}ms`);

                // Display context
                const contextHtml = response.sourceContext.map(ctx => `<p>${ctx}</p>`).join('');
                $('#contextDisplay').html(contextHtml);
            },
            error: function (xhr, status, error) {
                // Hide loading indicator
                $('.loading').hide();

                // Show error message
                $('.result-container').removeClass('d-none');
                $('#resultDisplay').text(`Error: ${error}`);
                $('#contextDisplay').html('');
            }
        });
    }
});