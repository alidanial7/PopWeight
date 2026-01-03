"""Correlation analysis workflow.

This module provides convenience wrappers for correlation analysis
between engagement metrics (Likes, Comments, Shares) and Reach.
"""

from analysis import analyze_correlation


def correlation_likes_reach() -> None:
    """
    Calculate and display correlation between Likes and Reach.

    This function analyzes the relationship between the number of likes
    a post receives and its reach, providing insights into how likes
    correlate with audience exposure.

    Examples
    --------
    >>> correlation_likes_reach()
    """
    analyze_correlation("Likes", "Reach", data_source="train")


def correlation_comments_reach() -> None:
    """
    Calculate and display correlation between Comments and Reach.

    This function analyzes the relationship between the number of comments
    a post receives and its reach, providing insights into how comments
    correlate with audience exposure.

    Examples
    --------
    >>> correlation_comments_reach()
    """
    analyze_correlation("Comments", "Reach", data_source="train")


def correlation_shares_reach() -> None:
    """
    Calculate and display correlation between Shares and Reach.

    This function analyzes the relationship between the number of shares
    a post receives and its reach, providing insights into how shares
    correlate with audience exposure.

    Examples
    --------
    >>> correlation_shares_reach()
    """
    analyze_correlation("Shares", "Reach", data_source="train")
