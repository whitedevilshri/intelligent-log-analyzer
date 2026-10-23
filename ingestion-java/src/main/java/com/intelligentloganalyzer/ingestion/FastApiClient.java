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

    private static final String DEFAULT_API_URL =
            "http://127.0.0.1:8000/api/v1/ingestion/batches";
    private static final String API_URL_PROPERTY = "intelligent.log.analyzer.api-url";
    private static final String API_URL_ENV = "INTELLIGENT_LOG_ANALYZER_API_URL";

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

        if (batchId == null || batchId.isBlank()) {
            throw new IllegalArgumentException("Batch ID must not be blank.");
        }

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
                .uri(resolveApiUri())
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

        JsonNode result;
        try {
            result = MAPPER.readTree(response.body());
        } catch (IOException exception) {
            throw new IOException(
                    "Batch response validation failed for " + batchId
                    + ": response body is not valid JSON.",
                    exception
            );
        }

        if (result == null
                || !result.isObject()
                || !"accepted".equals(result.path("status").asText())
                || !batchId.equals(result.path("batch_id").asText())
                || result.path("records_received").asInt(-1) != records.size()) {

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

    private static URI resolveApiUri() {
        String configuredUrl = System.getProperty(API_URL_PROPERTY);
        if (configuredUrl == null || configuredUrl.isBlank()) {
            configuredUrl = System.getenv(API_URL_ENV);
        }
        if (configuredUrl == null || configuredUrl.isBlank()) {
            configuredUrl = DEFAULT_API_URL;
        }

        try {
            return URI.create(configuredUrl);
        } catch (IllegalArgumentException exception) {
            throw new IllegalArgumentException(
                    "Invalid FastAPI endpoint configured by system property '"
                    + API_URL_PROPERTY + "' or environment variable '"
                    + API_URL_ENV + "': " + configuredUrl,
                    exception
            );
        }
    }
}
