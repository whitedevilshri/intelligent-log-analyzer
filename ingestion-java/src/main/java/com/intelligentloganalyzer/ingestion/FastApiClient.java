
package com.intelligentloganalyzer.ingestion;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;

public final class FastApiClient {

    private static final String API_URL =
            "http://127.0.0.1:8000/api/v1/ingestion/batches";

    private static final int MAX_BATCH_SIZE = 500;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final HttpClient CLIENT = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    private FastApiClient() {
    }

    public static void sendBatch(
            String batchId,
            List<Map<String, Object>> records
    ) throws IOException, InterruptedException {

        if (records == null || records.isEmpty()) {
            throw new IllegalArgumentException(
                    "Batch must contain at least one record."
            );
        }

        if (records.size() > MAX_BATCH_SIZE) {
            throw new IllegalArgumentException(
                    "Batch cannot contain more than 500 records."
            );
        }

        Map<String, Object> payload = Map.of(
                "batch_id", batchId,
                "records", records
        );

        String jsonBody = MAPPER.writeValueAsString(payload);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(API_URL))
                .version(HttpClient.Version.HTTP_1_1)
                .timeout(Duration.ofSeconds(30))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();

        HttpResponse<String> response = CLIENT.send(
                request,
                HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() != 202) {
            throw new IOException(
                    "Batch " + batchId
                    + " rejected. HTTP status: "
                    + response.statusCode()
                    + ", response: "
                    + response.body()
            );
        }

        JsonNode result = MAPPER.readTree(response.body());

        if (!"accepted".equals(result.path("status").asText())
                || !batchId.equals(result.path("batch_id").asText())
                || result.path("records_received").asInt(-1)
                != records.size()) {

            throw new IOException(
                    "Batch response validation failed for " + batchId
                    + ": " + response.body()
            );
        }

        System.out.println(
                "Batch accepted: " + batchId
                + " | Records: " + records.size()
                + " | HTTP: " + response.statusCode()
        );
    }
}
